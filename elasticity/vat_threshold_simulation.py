#!/usr/bin/env python3
"""
VAT Threshold & Rate Reform Simulation Model
Based on VAT_threshold_experiments_plan.md specification
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional, Literal
from pathlib import Path
import seaborn as sns

# Set style for better plots
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")


@dataclass
class FirmArchetype:
    """Core firm characteristics and behavioral parameters"""
    firm_id: str
    sic: str
    turnover_baseline: float  # Pre-policy turnover in £k
    b2b_share: float  # Share of sales to VAT-registered firms [0,1]
    input_vat_share: float  # Share of inputs bearing reclaimable VAT [0,1]
    gross_margin: float  # Pre-VAT gross margin [0,1]
    labor_intensity: Literal['low', 'med', 'high']
    region: Optional[str] = None
    
    # Behavioral parameters
    pass_through: float = 0.7  # Share of VAT change passed to consumer prices [0,1]
    demand_elasticity: float = -0.5  # Price elasticity of demand
    bunching_elasticity: float = 0.15  # Responsiveness of reported turnover near threshold
    bunching_band: float = 20.0  # Window around threshold for bunching (£k)


@dataclass
class PolicyScenario:
    """VAT policy configuration"""
    scenario_id: str
    scenario_name: str
    standard_rate: float  # e.g., 0.20 for 20%
    reduced_rate: Optional[float] = None  # For split-rate scenarios
    threshold_design: Literal['notch', 'taper'] = 'notch'
    
    # Threshold parameters
    T: Optional[float] = None  # Single threshold for notch (£k)
    T_start: Optional[float] = None  # Start of taper (£k)
    T_full: Optional[float] = None  # Full rate point for taper (£k)
    
    # Split-rate targeting
    split_rate_target: Optional[Dict] = None  # Rules for applying reduced rate
    
    def __post_init__(self):
        """Validate scenario configuration"""
        if self.threshold_design == 'notch':
            assert self.T is not None, "Notch design requires T"
        elif self.threshold_design == 'taper':
            assert self.T_start is not None and self.T_full is not None, "Taper requires T_start and T_full"
            assert self.T_full > self.T_start, "T_full must be greater than T_start"


@dataclass
class SimulationResult:
    """Results for a single firm under a scenario"""
    firm_id: str
    scenario_id: str
    
    # Pre-policy state
    turnover_baseline: float
    
    # Post-policy state
    turnover_reported: float  # After bunching behavior
    vat_liability: float
    effective_vat_rate: float
    operating_profit_change: float
    registered: bool
    
    # Decomposition
    static_vat: float  # VAT without behavioral response
    behavioral_vat: float  # VAT with behavioral response
    quantity_change: float  # % change in quantity sold
    price_change: float  # % change in price
    
    # Metrics
    bunching_adjustment: float  # Amount of turnover reduction due to bunching


class VATSimulator:
    """Main simulation engine for VAT threshold analysis"""
    
    def __init__(self, mode: Literal['static', 'behavioural'] = 'behavioural'):
        self.mode = mode
        self.results: List[SimulationResult] = []
        
    def calculate_vat_liability(self, firm: FirmArchetype, scenario: PolicyScenario, 
                               turnover: float, registered: bool) -> float:
        """Calculate VAT liability for a firm under a scenario"""
        
        # Determine applicable VAT rate
        vat_rate = scenario.standard_rate
        if scenario.reduced_rate is not None and scenario.split_rate_target is not None:
            # Check if firm qualifies for reduced rate
            if self._qualifies_for_reduced_rate(firm, scenario.split_rate_target):
                vat_rate = scenario.reduced_rate
        
        if not registered:
            return 0.0
        
        # Calculate output VAT (only on B2C portion)
        b2c_share = 1 - firm.b2b_share
        output_vat = turnover * b2c_share * vat_rate
        
        # Calculate input VAT credit (approximation)
        costs = turnover * (1 - firm.gross_margin)
        input_vat_credit = costs * firm.input_vat_share * scenario.standard_rate
        
        # Net VAT liability
        net_vat = max(0, output_vat - input_vat_credit)
        
        # Apply taper if applicable
        if scenario.threshold_design == 'taper' and scenario.T_start <= turnover < scenario.T_full:
            taper_factor = (turnover - scenario.T_start) / (scenario.T_full - scenario.T_start)
            net_vat *= taper_factor
            
        return net_vat
    
    def _qualifies_for_reduced_rate(self, firm: FirmArchetype, split_rate_target: Dict) -> bool:
        """Check if firm qualifies for reduced VAT rate"""
        if 'labor_intensity' in split_rate_target:
            if firm.labor_intensity != split_rate_target['labor_intensity']:
                return False
        if 'sic_list' in split_rate_target:
            if firm.sic not in split_rate_target['sic_list']:
                return False
        return True
    
    def _determine_registration(self, turnover: float, scenario: PolicyScenario) -> bool:
        """Determine if firm must register for VAT"""
        if scenario.threshold_design == 'notch':
            return turnover >= scenario.T
        elif scenario.threshold_design == 'taper':
            return turnover >= scenario.T_start
        return False
    
    def _apply_bunching_behavior(self, firm: FirmArchetype, scenario: PolicyScenario) -> float:
        """Apply bunching behavior near threshold"""
        if self.mode == 'static':
            return firm.turnover_baseline
        
        turnover = firm.turnover_baseline
        
        # Determine relevant threshold
        if scenario.threshold_design == 'notch':
            threshold = scenario.T
        elif scenario.threshold_design == 'taper':
            threshold = scenario.T_start
        else:
            return turnover
        
        # Check if firm is in bunching band
        if threshold - firm.bunching_band <= turnover <= threshold + firm.bunching_band:
            # Calculate bunching adjustment
            distance_to_threshold = turnover - threshold
            
            if distance_to_threshold > 0:  # Above threshold
                # Firms above threshold might reduce turnover to just below
                adjustment_strength = firm.bunching_elasticity
                
                # Taper reduces bunching incentive
                if scenario.threshold_design == 'taper':
                    taper_intensity = 1 / (scenario.T_full - scenario.T_start) * 100
                    adjustment_strength *= max(0, 1 - taper_intensity)
                
                # Calculate adjustment
                max_adjustment = distance_to_threshold + 0.5  # Push to just below threshold
                actual_adjustment = min(max_adjustment, adjustment_strength * distance_to_threshold * 10)
                
                turnover = max(threshold - 0.5, turnover - actual_adjustment)
        
        return turnover
    
    def _apply_demand_response(self, firm: FirmArchetype, scenario: PolicyScenario,
                              registered: bool) -> Tuple[float, float]:
        """Calculate price and quantity changes due to VAT"""
        if self.mode == 'static' or not registered:
            return 0.0, 0.0
        
        # Determine VAT rate change (assuming baseline is current 20% rate)
        baseline_rate = 0.20
        
        new_rate = scenario.standard_rate
        if scenario.reduced_rate is not None and scenario.split_rate_target is not None:
            if self._qualifies_for_reduced_rate(firm, scenario.split_rate_target):
                new_rate = scenario.reduced_rate
        
        vat_rate_change = new_rate - baseline_rate
        
        # Only B2C sales affected by price changes
        b2c_share = 1 - firm.b2b_share
        
        # Price change
        price_change = firm.pass_through * vat_rate_change * b2c_share
        
        # Quantity change
        quantity_change = firm.demand_elasticity * price_change
        
        return price_change, quantity_change
    
    def simulate_firm(self, firm: FirmArchetype, scenario: PolicyScenario) -> SimulationResult:
        """Simulate a single firm under a scenario"""
        
        # Apply bunching behavior
        turnover_reported = self._apply_bunching_behavior(firm, scenario)
        bunching_adjustment = firm.turnover_baseline - turnover_reported
        
        # Determine registration status
        registered = self._determine_registration(turnover_reported, scenario)
        
        # Calculate static VAT liability
        static_vat = self.calculate_vat_liability(firm, scenario, firm.turnover_baseline, registered)
        
        # Apply demand response
        price_change, quantity_change = self._apply_demand_response(firm, scenario, registered)
        
        # Adjust turnover for quantity change
        if self.mode == 'behavioural':
            turnover_final = turnover_reported * (1 + quantity_change)
        else:
            turnover_final = turnover_reported
        
        # Calculate behavioral VAT liability
        behavioral_vat = self.calculate_vat_liability(firm, scenario, turnover_final, registered)
        
        # Calculate effective VAT rate
        if turnover_final > 0:
            effective_vat_rate = behavioral_vat / turnover_final
        else:
            effective_vat_rate = 0.0
        
        # Calculate profit change
        revenue_change = (turnover_final - firm.turnover_baseline) * firm.gross_margin
        operating_profit_change = revenue_change - behavioral_vat
        
        return SimulationResult(
            firm_id=firm.firm_id,
            scenario_id=scenario.scenario_id,
            turnover_baseline=firm.turnover_baseline,
            turnover_reported=turnover_final,
            vat_liability=behavioral_vat,
            effective_vat_rate=effective_vat_rate,
            operating_profit_change=operating_profit_change,
            registered=registered,
            static_vat=static_vat,
            behavioral_vat=behavioral_vat,
            quantity_change=quantity_change,
            price_change=price_change,
            bunching_adjustment=bunching_adjustment
        )
    
    def simulate_all(self, firms: List[FirmArchetype], 
                    scenarios: List[PolicyScenario]) -> pd.DataFrame:
        """Run simulations for all firm-scenario combinations"""
        
        self.results = []
        
        for scenario in scenarios:
            for firm in firms:
                result = self.simulate_firm(firm, scenario)
                self.results.append(result)
        
        # Convert to DataFrame
        df = pd.DataFrame([vars(r) for r in self.results])
        return df


def create_scenarios() -> List[PolicyScenario]:
    """Create all policy scenarios from specification"""
    
    scenarios = [
        PolicyScenario(
            scenario_id='S0',
            scenario_name='Baseline',
            standard_rate=0.20,
            threshold_design='notch',
            T=90.0
        ),
        PolicyScenario(
            scenario_id='S1',
            scenario_name='Higher threshold',
            standard_rate=0.20,
            threshold_design='notch',
            T=100.0
        ),
        PolicyScenario(
            scenario_id='S2',
            scenario_name='Taper A (£65k-£110k)',
            standard_rate=0.20,
            threshold_design='taper',
            T_start=65.0,
            T_full=110.0
        ),
        PolicyScenario(
            scenario_id='S3',
            scenario_name='Taper B (£90k-£135k)',
            standard_rate=0.20,
            threshold_design='taper',
            T_start=90.0,
            T_full=135.0
        ),
        PolicyScenario(
            scenario_id='S4',
            scenario_name='Split-rate',
            standard_rate=0.20,
            reduced_rate=0.10,
            threshold_design='notch',
            T=90.0,
            split_rate_target={'labor_intensity': 'high'}
        ),
        PolicyScenario(
            scenario_id='S5',
            scenario_name='Split-rate + Taper B',
            standard_rate=0.20,
            reduced_rate=0.10,
            threshold_design='taper',
            T_start=90.0,
            T_full=135.0,
            split_rate_target={'labor_intensity': 'high'}
        )
    ]
    
    return scenarios


def create_firm_archetypes() -> List[FirmArchetype]:
    """Create firm archetypes from specification"""
    
    firms = []
    
    # A1: Hairdresser (B2C, high labor)
    for turnover in [80, 89, 91, 105]:
        firms.append(FirmArchetype(
            firm_id=f'A1_hairdresser_{turnover}k',
            sic='96020',
            turnover_baseline=turnover,
            b2b_share=0.05,
            input_vat_share=0.2,
            gross_margin=0.45,
            labor_intensity='high',
            pass_through=0.7,
            demand_elasticity=-0.4,
            bunching_elasticity=0.20,
            bunching_band=20.0
        ))
    
    # A2: Bicycle repair (B2C, high labor)
    for turnover in [80, 89, 91, 105]:
        firms.append(FirmArchetype(
            firm_id=f'A2_bicycle_{turnover}k',
            sic='95290',
            turnover_baseline=turnover,
            b2b_share=0.05,
            input_vat_share=0.2,
            gross_margin=0.40,
            labor_intensity='high',
            pass_through=0.6,
            demand_elasticity=-0.5,
            bunching_elasticity=0.15,
            bunching_band=20.0
        ))
    
    # A3: Cafe (B2C, medium labor, higher inputs)
    for turnover in [85, 95, 120]:
        firms.append(FirmArchetype(
            firm_id=f'A3_cafe_{turnover}k',
            sic='56101',
            turnover_baseline=turnover,
            b2b_share=0.10,
            input_vat_share=0.4,
            gross_margin=0.35,
            labor_intensity='med',
            pass_through=0.8,
            demand_elasticity=-0.8,
            bunching_elasticity=0.10,
            bunching_band=20.0
        ))
    
    # A4: Small contractor (mixed B2B/B2C)
    for turnover in [88, 100, 130]:
        firms.append(FirmArchetype(
            firm_id=f'A4_contractor_{turnover}k',
            sic='41201',
            turnover_baseline=turnover,
            b2b_share=0.6,
            input_vat_share=0.3,
            gross_margin=0.25,
            labor_intensity='med',
            pass_through=0.5,
            demand_elasticity=-0.3,
            bunching_elasticity=0.10,
            bunching_band=20.0
        ))
    
    # A5: B2B services (accounting)
    for turnover in [85, 95, 140]:
        firms.append(FirmArchetype(
            firm_id=f'A5_accounting_{turnover}k',
            sic='69201',
            turnover_baseline=turnover,
            b2b_share=0.95,
            input_vat_share=0.2,
            gross_margin=0.60,
            labor_intensity='high',
            pass_through=0.3,
            demand_elasticity=-0.2,
            bunching_elasticity=0.05,
            bunching_band=20.0
        ))
    
    return firms


def calculate_metrics(df: pd.DataFrame) -> Dict:
    """Calculate key metrics from simulation results"""
    
    metrics = {}
    
    for scenario_id in df['scenario_id'].unique():
        scenario_df = df[df['scenario_id'] == scenario_id]
        
        metrics[scenario_id] = {
            'total_vat_receipts': scenario_df['vat_liability'].sum(),
            'num_registered': scenario_df['registered'].sum(),
            'avg_effective_rate': scenario_df['effective_vat_rate'].mean(),
            'total_profit_change': scenario_df['operating_profit_change'].sum(),
            'avg_bunching': scenario_df['bunching_adjustment'].mean(),
            'max_bunching': scenario_df['bunching_adjustment'].max()
        }
    
    return metrics


def plot_profit_continuity(df: pd.DataFrame, scenarios_to_plot: List[str] = ['S0', 'S2', 'S3']):
    """Plot after-tax profit vs turnover around threshold"""
    
    fig, axes = plt.subplots(1, len(scenarios_to_plot), figsize=(15, 5))
    
    for idx, scenario_id in enumerate(scenarios_to_plot):
        ax = axes[idx] if len(scenarios_to_plot) > 1 else axes
        
        # Create synthetic data points for smooth curve
        turnovers = np.linspace(60, 150, 200)
        profits = []
        
        # Get a representative firm (use hairdresser as example)
        base_firm = FirmArchetype(
            firm_id='test',
            sic='96020',
            turnover_baseline=0,  # Will be overridden
            b2b_share=0.05,
            input_vat_share=0.2,
            gross_margin=0.45,
            labor_intensity='high',
            pass_through=0.7,
            demand_elasticity=-0.4,
            bunching_elasticity=0.20,
            bunching_band=20.0
        )
        
        scenario = [s for s in create_scenarios() if s.scenario_id == scenario_id][0]
        simulator = VATSimulator(mode='static')  # Static for cleaner visualization
        
        for t in turnovers:
            base_firm.turnover_baseline = t
            result = simulator.simulate_firm(base_firm, scenario)
            profit = t * base_firm.gross_margin - result.vat_liability
            profits.append(profit)
        
        ax.plot(turnovers, profits, linewidth=2)
        ax.set_xlabel('Turnover (£k)')
        ax.set_ylabel('After-tax Profit (£k)')
        ax.set_title(f'{scenario_id}: {scenario.scenario_name}')
        ax.grid(True, alpha=0.3)
        
        # Add threshold markers
        if scenario.threshold_design == 'notch':
            ax.axvline(x=scenario.T, color='red', linestyle='--', alpha=0.5)
            ax.text(scenario.T, ax.get_ylim()[1]*0.95, f'T={scenario.T}k', 
                   ha='center', color='red')
        elif scenario.threshold_design == 'taper':
            ax.axvline(x=scenario.T_start, color='orange', linestyle='--', alpha=0.5)
            ax.axvline(x=scenario.T_full, color='orange', linestyle='--', alpha=0.5)
            ax.axvspan(scenario.T_start, scenario.T_full, alpha=0.1, color='orange')
            ax.text((scenario.T_start + scenario.T_full)/2, ax.get_ylim()[1]*0.95, 
                   'Taper zone', ha='center', color='orange')
    
    plt.suptitle('Profit Continuity Analysis: Notch vs Taper Designs')
    plt.tight_layout()
    return fig


def plot_bunching_distribution(df: pd.DataFrame):
    """Plot turnover distribution showing bunching effects"""
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()
    
    for idx, scenario_id in enumerate(['S0', 'S1', 'S2', 'S3', 'S4', 'S5']):
        ax = axes[idx]
        scenario_df = df[df['scenario_id'] == scenario_id]
        
        # Create histogram of reported turnover
        ax.hist(scenario_df['turnover_reported'], bins=20, alpha=0.7, color='blue', 
                edgecolor='black', label='Reported')
        ax.hist(scenario_df['turnover_baseline'], bins=20, alpha=0.5, color='gray', 
                edgecolor='black', label='Baseline')
        
        ax.set_xlabel('Turnover (£k)')
        ax.set_ylabel('Number of Firms')
        ax.set_title(f'{scenario_id}')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Add threshold marker
        scenario = [s for s in create_scenarios() if s.scenario_id == scenario_id][0]
        if scenario.threshold_design == 'notch':
            ax.axvline(x=scenario.T, color='red', linestyle='--', alpha=0.7)
        elif scenario.threshold_design == 'taper':
            ax.axvspan(scenario.T_start, scenario.T_full, alpha=0.2, color='orange')
    
    plt.suptitle('Turnover Distribution and Bunching Effects')
    plt.tight_layout()
    return fig


def create_summary_table(df: pd.DataFrame) -> pd.DataFrame:
    """Create summary table of key results"""
    
    summary = []
    
    for scenario_id in df['scenario_id'].unique():
        scenario_df = df[df['scenario_id'] == scenario_id]
        
        # Group by firm type
        for firm_type in ['hairdresser', 'bicycle', 'cafe', 'contractor', 'accounting']:
            type_df = scenario_df[scenario_df['firm_id'].str.contains(firm_type)]
            
            if len(type_df) > 0:
                summary.append({
                    'Scenario': scenario_id,
                    'Firm Type': firm_type.capitalize(),
                    'Avg VAT (£k)': type_df['vat_liability'].mean(),
                    'Avg Effective Rate': type_df['effective_vat_rate'].mean(),
                    'Registered (%)': type_df['registered'].mean() * 100,
                    'Avg Profit Change (£k)': type_df['operating_profit_change'].mean(),
                    'Avg Bunching (£k)': type_df['bunching_adjustment'].mean()
                })
    
    return pd.DataFrame(summary)


def sensitivity_analysis(base_firm: FirmArchetype, scenario: PolicyScenario,
                        param_ranges: Dict) -> pd.DataFrame:
    """Perform sensitivity analysis on key parameters"""
    
    results = []
    simulator = VATSimulator(mode='behavioural')
    
    # Base case
    base_result = simulator.simulate_firm(base_firm, scenario)
    base_vat = base_result.vat_liability
    
    # Vary each parameter
    for param_name, values in param_ranges.items():
        for value in values:
            # Create modified firm
            test_firm = FirmArchetype(**vars(base_firm))
            setattr(test_firm, param_name, value)
            
            # Simulate
            result = simulator.simulate_firm(test_firm, scenario)
            
            results.append({
                'Parameter': param_name,
                'Value': value,
                'VAT Liability': result.vat_liability,
                'VAT Change from Base': result.vat_liability - base_vat,
                'Bunching': result.bunching_adjustment,
                'Effective Rate': result.effective_vat_rate
            })
    
    return pd.DataFrame(results)


def main():
    """Run full simulation suite"""
    
    print("=" * 60)
    print("VAT THRESHOLD SIMULATION MODEL")
    print("=" * 60)
    
    # Create scenarios and firms
    scenarios = create_scenarios()
    firms = create_firm_archetypes()
    
    print(f"\nCreated {len(scenarios)} scenarios and {len(firms)} firm archetypes")
    
    # Run simulations
    print("\nRunning simulations...")
    
    # Static mode
    static_simulator = VATSimulator(mode='static')
    static_results = static_simulator.simulate_all(firms, scenarios)
    static_results['mode'] = 'static'
    
    # Behavioral mode
    behavioral_simulator = VATSimulator(mode='behavioural')
    behavioral_results = behavioral_simulator.simulate_all(firms, scenarios)
    behavioral_results['mode'] = 'behavioural'
    
    # Combine results
    all_results = pd.concat([static_results, behavioral_results], ignore_index=True)
    
    print(f"Completed {len(all_results)} simulation runs")
    
    # Calculate metrics
    print("\n" + "=" * 60)
    print("KEY METRICS BY SCENARIO (Behavioural Mode)")
    print("=" * 60)
    
    behavioral_metrics = calculate_metrics(behavioral_results)
    
    for scenario_id, metrics in behavioral_metrics.items():
        scenario = [s for s in scenarios if s.scenario_id == scenario_id][0]
        print(f"\n{scenario_id}: {scenario.scenario_name}")
        print("-" * 40)
        print(f"  Total VAT Receipts: £{metrics['total_vat_receipts']:.2f}k")
        print(f"  Firms Registered: {metrics['num_registered']:.0f}")
        print(f"  Avg Effective Rate: {metrics['avg_effective_rate']:.1%}")
        print(f"  Total Profit Change: £{metrics['total_profit_change']:.2f}k")
        print(f"  Avg Bunching: £{metrics['avg_bunching']:.2f}k")
        print(f"  Max Bunching: £{metrics['max_bunching']:.2f}k")
    
    # Create summary table
    print("\n" + "=" * 60)
    print("SUMMARY BY FIRM TYPE")
    print("=" * 60)
    
    summary_table = create_summary_table(behavioral_results)
    print(summary_table.to_string(index=False))
    
    # Sensitivity analysis for hairdresser
    print("\n" + "=" * 60)
    print("SENSITIVITY ANALYSIS: Hairdresser at £89k")
    print("=" * 60)
    
    test_firm = [f for f in firms if f.firm_id == 'A1_hairdresser_89k'][0]
    baseline_scenario = scenarios[0]  # S0
    
    param_ranges = {
        'pass_through': [0.4, 0.7, 1.0],
        'demand_elasticity': [-0.3, -0.6, -0.9],
        'bunching_elasticity': [0.05, 0.15, 0.30]
    }
    
    sensitivity_df = sensitivity_analysis(test_firm, baseline_scenario, param_ranges)
    print(sensitivity_df.to_string(index=False))
    
    # Generate plots
    print("\n" + "=" * 60)
    print("GENERATING VISUALIZATIONS")
    print("=" * 60)
    
    # Save results to CSV
    output_dir = Path(__file__).parent / 'results'
    output_dir.mkdir(exist_ok=True)
    
    all_results.to_csv(output_dir / 'simulation_results.csv', index=False)
    summary_table.to_csv(output_dir / 'summary_by_firm_type.csv', index=False)
    sensitivity_df.to_csv(output_dir / 'sensitivity_analysis.csv', index=False)
    
    print(f"\nResults saved to {output_dir}")
    
    # Create plots
    fig1 = plot_profit_continuity(behavioral_results)
    fig1.savefig(output_dir / 'profit_continuity.png', dpi=300, bbox_inches='tight')
    
    fig2 = plot_bunching_distribution(behavioral_results)
    fig2.savefig(output_dir / 'bunching_distribution.png', dpi=300, bbox_inches='tight')
    
    print("Plots saved to results directory")
    
    # Validation checks
    print("\n" + "=" * 60)
    print("VALIDATION CHECKS")
    print("=" * 60)
    
    # Check 1: Continuity improvement with tapers
    s0_metrics = behavioral_metrics['S0']
    s2_metrics = behavioral_metrics['S2']
    s3_metrics = behavioral_metrics['S3']
    
    print("\n✓ Bunching reduction with tapers:")
    print(f"  S0 (notch) max bunching: £{s0_metrics['max_bunching']:.2f}k")
    print(f"  S2 (taper A) max bunching: £{s2_metrics['max_bunching']:.2f}k")
    print(f"  S3 (taper B) max bunching: £{s3_metrics['max_bunching']:.2f}k")
    
    assert s2_metrics['max_bunching'] < s0_metrics['max_bunching'], "Taper should reduce bunching"
    assert s3_metrics['max_bunching'] < s0_metrics['max_bunching'], "Taper should reduce bunching"
    
    # Check 2: B2B vs B2C incidence
    b2b_firms = behavioral_results[behavioral_results['firm_id'].str.contains('accounting')]
    b2c_firms = behavioral_results[behavioral_results['firm_id'].str.contains('hairdresser')]
    
    b2b_avg_rate = b2b_firms['effective_vat_rate'].mean()
    b2c_avg_rate = b2c_firms['effective_vat_rate'].mean()
    
    print("\n✓ B2B vs B2C VAT incidence:")
    print(f"  B2B (accounting) avg rate: {b2b_avg_rate:.2%}")
    print(f"  B2C (hairdresser) avg rate: {b2c_avg_rate:.2%}")
    
    assert b2b_avg_rate < b2c_avg_rate, "B2B firms should have lower effective rates"
    
    print("\n" + "=" * 60)
    print("SIMULATION COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    main()