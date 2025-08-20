#!/usr/bin/env python3
"""
Enhanced VAT Threshold Simulation Model
Integrates synthetic UK firm data with empirical parameters
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional, Literal
from pathlib import Path
import seaborn as sns
from scipy import stats
from scipy.optimize import minimize_scalar

# Set style for better plots
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")


# ============================================================================
# EMPIRICAL PARAMETERS FROM LITERATURE
# ============================================================================

EMPIRICAL_ELASTICITIES = {
    'bunching': {
        'default': 0.15,
        '47': 0.17,  # Retail - Harju & Matikka (2016)
        '56': 0.23,  # Food service - Best et al. (2015)
        '41': 0.12,  # Construction - Kleven & Waseem (2013)
        '96': 0.20,  # Personal services
        '69': 0.08,  # Professional services (less responsive)
    },
    'demand': {
        'default': -0.6,
        '56': -1.2,  # Restaurants - Harding et al. (2012)
        '96': -0.4,  # Hairdressing - Davis (2011)
        '47': -0.8,  # Retail - Benedek et al. (2020)
        '69': -0.2,  # Professional services (inelastic)
        '41': -0.3,  # Construction (intermediate)
    },
    'pass_through': {
        'default': 0.6,
        'competitive': 0.7,  # Benedek et al. (2020)
        'services': 0.5,     # Kosonen (2015)
        'monopolistic': 0.3  # Carbonnier (2007)
    }
}

VAT_COMPLIANCE_COSTS = {
    'initial_registration': 1.5,  # £k one-time
    'annual_compliance': 3.0,     # £k per year
    'per_return': 0.15,          # £k quarterly (0.6k annual)
    'software_costs': 0.5,        # £k annual
    'accountant_fees': {
        'micro': 1.2,   # <50k turnover
        'small': 3.6,   # 50-150k
        'medium': 8.0   # 150k+
    }
}

FLAT_RATE_SCHEME = {
    # Sector-specific flat rates (simplified)
    '41': 0.095,  # Construction
    '47': 0.075,  # Retail
    '56': 0.125,  # Catering
    '69': 0.145,  # Professional services
    '96': 0.130,  # Personal services
    'default': 0.12,
    'threshold': 150.0,  # £k
    'exit_threshold': 230.0  # £k
}

ENFORCEMENT_PARAMETERS = {
    'audit_probability': {
        'below_threshold': 0.02,
        'at_threshold': 0.08,  # Higher scrutiny near threshold
        'above_threshold': 0.04
    },
    'penalty_rates': {
        'late_registration': 0.05,
        'incorrect_return': 0.15,
        'deliberate_evasion': 0.70
    },
    'detection_lag_years': 2
}

CASH_FLOW_PARAMETERS = {
    'payment_terms': {
        'b2b_days': 45,
        'b2c_days': 0,
        'vat_payment_days': 37  # After quarter end
    },
    'working_capital_cost': 0.08,  # Annual rate
    'bad_debt_rate': 0.02
}


# ============================================================================
# ENHANCED DATA STRUCTURES
# ============================================================================

@dataclass
class EnhancedFirm:
    """Enhanced firm with compliance costs and scheme options"""
    firm_id: str
    sic: str
    turnover_baseline: float
    employment: int
    b2b_share: float
    input_vat_share: float
    gross_margin: float
    weight: float  # Population weight
    
    # Behavioral parameters (will be set from empirical data)
    pass_through: float = 0.6
    demand_elasticity: float = -0.5
    bunching_elasticity: float = 0.15
    
    # VAT scheme choices
    flat_rate_eligible: bool = False
    cash_accounting: bool = False
    
    # Compliance burden
    compliance_capacity: Literal['low', 'medium', 'high'] = 'medium'
    
    def get_compliance_cost(self, registered: bool) -> float:
        """Calculate annual compliance cost"""
        if not registered:
            return 0
        
        # Size-based fees
        if self.turnover_baseline < 50:
            accountant_fee = VAT_COMPLIANCE_COSTS['accountant_fees']['micro']
        elif self.turnover_baseline < 150:
            accountant_fee = VAT_COMPLIANCE_COSTS['accountant_fees']['small']
        else:
            accountant_fee = VAT_COMPLIANCE_COSTS['accountant_fees']['medium']
        
        annual_cost = (
            VAT_COMPLIANCE_COSTS['annual_compliance'] +
            VAT_COMPLIANCE_COSTS['per_return'] * 4 +  # Quarterly
            VAT_COMPLIANCE_COSTS['software_costs'] +
            accountant_fee
        )
        
        # Adjust for compliance capacity
        if self.compliance_capacity == 'low':
            annual_cost *= 1.3  # Higher burden for less sophisticated firms
        elif self.compliance_capacity == 'high':
            annual_cost *= 0.7  # Lower burden for sophisticated firms
            
        return annual_cost


@dataclass
class PolicyScenarioEnhanced:
    """Enhanced policy scenario with scheme options"""
    scenario_id: str
    scenario_name: str
    standard_rate: float
    reduced_rate: Optional[float] = None
    threshold_design: Literal['notch', 'taper'] = 'notch'
    
    # Threshold parameters
    T: Optional[float] = None
    T_start: Optional[float] = None
    T_full: Optional[float] = None
    
    # Scheme options
    allow_flat_rate: bool = True
    flat_rate_changes: Optional[Dict] = None
    
    # Enforcement
    enforcement_intensity: float = 1.0  # Multiplier on audit probabilities
    
    # Split-rate targeting
    split_rate_target: Optional[Dict] = None


# ============================================================================
# EMPIRICAL ANALYSIS FUNCTIONS
# ============================================================================

def analyze_bunching_in_data(df: pd.DataFrame, threshold: float = 90.0, 
                            bandwidth: float = 20.0) -> Dict:
    """Extract empirical bunching patterns from synthetic data"""
    
    # Filter to relevant range
    bunching_df = df[(df['annual_turnover_k'] >= threshold - bandwidth) & 
                     (df['annual_turnover_k'] <= threshold + bandwidth)]
    
    # Calculate bunching mass
    below_threshold = bunching_df[bunching_df['annual_turnover_k'] < threshold]['weight'].sum()
    above_threshold = bunching_df[bunching_df['annual_turnover_k'] >= threshold]['weight'].sum()
    
    # Estimate counterfactual (polynomial fit excluding bunching region)
    exclude_region = (threshold - 5, threshold + 2)
    fit_data = df[(df['annual_turnover_k'] >= threshold - bandwidth) & 
                  (df['annual_turnover_k'] <= threshold + bandwidth)]
    fit_data = fit_data[(fit_data['annual_turnover_k'] < exclude_region[0]) | 
                        (fit_data['annual_turnover_k'] > exclude_region[1])]
    
    if len(fit_data) > 10:
        # Fit polynomial
        z = np.polyfit(fit_data['annual_turnover_k'], fit_data['weight'], 2)
        p = np.poly1d(z)
        
        # Calculate excess mass
        actual = bunching_df[(bunching_df['annual_turnover_k'] >= threshold - 5) & 
                            (bunching_df['annual_turnover_k'] < threshold)]['weight'].sum()
        counterfactual = sum([p(x) for x in range(int(threshold - 5), int(threshold))])
        excess_mass = actual - counterfactual
    else:
        excess_mass = 0
    
    # Calculate by sector
    sector_bunching = {}
    for sic in df['sic_code'].unique():
        sector_df = df[df['sic_code'] == sic]
        sector_below = sector_df[(sector_df['annual_turnover_k'] >= threshold - 5) & 
                                (sector_df['annual_turnover_k'] < threshold)]['weight'].sum()
        sector_above = sector_df[(sector_df['annual_turnover_k'] >= threshold) & 
                                (sector_df['annual_turnover_k'] < threshold + 5)]['weight'].sum()
        
        if sector_above > 0:
            sector_bunching[sic] = (sector_below - sector_above) / sector_above
        else:
            sector_bunching[sic] = 0
    
    return {
        'total_excess_mass': excess_mass,
        'bunching_ratio': below_threshold / above_threshold if above_threshold > 0 else 0,
        'sector_bunching': sector_bunching,
        'below_threshold_mass': below_threshold,
        'above_threshold_mass': above_threshold
    }


def estimate_sector_parameters(df: pd.DataFrame) -> Dict:
    """Estimate behavioral parameters by sector from data"""
    
    bunching_analysis = analyze_bunching_in_data(df)
    
    sector_params = {}
    for sic in df['sic_code'].unique():
        sic_str = str(sic)
        
        # Get bunching elasticity from empirical estimates or data
        if sic_str in EMPIRICAL_ELASTICITIES['bunching']:
            bunching_e = EMPIRICAL_ELASTICITIES['bunching'][sic_str]
        else:
            # Estimate from bunching mass
            observed_bunching = bunching_analysis['sector_bunching'].get(sic, 0)
            bunching_e = min(0.3, max(0.05, observed_bunching * 0.5))  # Scale to reasonable range
        
        # Get demand elasticity
        if sic_str in EMPIRICAL_ELASTICITIES['demand']:
            demand_e = EMPIRICAL_ELASTICITIES['demand'][sic_str]
        else:
            demand_e = EMPIRICAL_ELASTICITIES['demand']['default']
        
        # Estimate pass-through based on market structure
        sector_df = df[df['sic_code'] == sic]
        firm_count = len(sector_df)
        
        if firm_count > 1000:  # Competitive
            pass_through = 0.7
        elif firm_count > 100:  # Moderate competition
            pass_through = 0.5
        else:  # Less competitive
            pass_through = 0.3
        
        sector_params[sic] = {
            'bunching_elasticity': bunching_e,
            'demand_elasticity': demand_e,
            'pass_through': pass_through,
            'avg_margin': sector_df['annual_turnover_k'].std() / sector_df['annual_turnover_k'].mean() if len(sector_df) > 0 else 0.3
        }
    
    return sector_params


# ============================================================================
# ENHANCED SIMULATION ENGINE
# ============================================================================

class EnhancedVATSimulator:
    """Enhanced VAT simulation with empirical grounding"""
    
    def __init__(self, synthetic_data_path: Optional[str] = None):
        """Initialize with synthetic data if available"""
        
        self.synthetic_df = None
        self.sector_params = None
        self.bunching_patterns = None
        
        if synthetic_data_path:
            self.load_synthetic_data(synthetic_data_path)
    
    def load_synthetic_data(self, path: str):
        """Load and analyze synthetic firm data"""
        
        print("Loading synthetic firm data...")
        self.synthetic_df = pd.read_csv(path)
        
        # Analyze bunching patterns
        print("Analyzing bunching patterns...")
        self.bunching_patterns = analyze_bunching_in_data(self.synthetic_df)
        
        # Estimate sector parameters
        print("Estimating sector parameters...")
        self.sector_params = estimate_sector_parameters(self.synthetic_df)
        
        print(f"Loaded {len(self.synthetic_df):,} firms")
        print(f"Empirical bunching ratio: {self.bunching_patterns['bunching_ratio']:.2f}")
    
    def create_firm_sample(self, n_firms: int = 10000, 
                          turnover_range: Tuple[float, float] = (70, 120)) -> List[EnhancedFirm]:
        """Create realistic firm sample from synthetic data"""
        
        if self.synthetic_df is None:
            raise ValueError("Must load synthetic data first")
        
        # Sample firms in relevant range
        relevant_df = self.synthetic_df[
            (self.synthetic_df['annual_turnover_k'] >= turnover_range[0]) &
            (self.synthetic_df['annual_turnover_k'] <= turnover_range[1])
        ]
        
        if len(relevant_df) < n_firms:
            sample_df = relevant_df
        else:
            sample_df = relevant_df.sample(n=n_firms, weights='weight')
        
        firms = []
        for idx, row in sample_df.iterrows():
            sic = row['sic_code']
            
            # Get sector-specific parameters
            if sic in self.sector_params:
                params = self.sector_params[sic]
            else:
                params = {
                    'bunching_elasticity': 0.15,
                    'demand_elasticity': -0.6,
                    'pass_through': 0.6,
                    'avg_margin': 0.3
                }
            
            # Estimate B2B share based on sector
            if sic in [69, 70, 71, 73, 74]:  # Professional services
                b2b_share = np.random.beta(8, 2)  # High B2B
            elif sic in [47, 56, 96]:  # Retail, food, personal services
                b2b_share = np.random.beta(2, 8)  # Low B2B
            else:
                b2b_share = np.random.beta(5, 5)  # Mixed
            
            # Estimate input VAT share
            input_vat_share = np.random.beta(3, 2) * 0.5  # Usually 20-40%
            
            # Determine compliance capacity
            if row['employment'] < 5:
                compliance_capacity = 'low'
            elif row['employment'] < 20:
                compliance_capacity = 'medium'
            else:
                compliance_capacity = 'high'
            
            firm = EnhancedFirm(
                firm_id=f"firm_{idx}",
                sic=str(sic),
                turnover_baseline=row['annual_turnover_k'],
                employment=int(row['employment']),
                b2b_share=b2b_share,
                input_vat_share=input_vat_share,
                gross_margin=params['avg_margin'],
                weight=row['weight'],
                pass_through=params['pass_through'],
                demand_elasticity=params['demand_elasticity'],
                bunching_elasticity=params['bunching_elasticity'],
                compliance_capacity=compliance_capacity
            )
            
            # Check flat rate eligibility
            firm.flat_rate_eligible = firm.turnover_baseline <= FLAT_RATE_SCHEME['threshold']
            
            firms.append(firm)
        
        return firms
    
    def calculate_vat_liability_enhanced(self, firm: EnhancedFirm, 
                                        scenario: PolicyScenarioEnhanced,
                                        turnover: float, 
                                        registered: bool) -> Tuple[float, str]:
        """Calculate VAT with scheme options"""
        
        if not registered:
            return 0.0, 'unregistered'
        
        # Determine if using flat rate scheme
        if (firm.flat_rate_eligible and 
            scenario.allow_flat_rate and 
            turnover <= FLAT_RATE_SCHEME['threshold']):
            
            # Get sector-specific flat rate
            if firm.sic in FLAT_RATE_SCHEME:
                flat_rate = FLAT_RATE_SCHEME[firm.sic]
            else:
                flat_rate = FLAT_RATE_SCHEME['default']
            
            # Apply any policy changes to flat rate
            if scenario.flat_rate_changes and firm.sic in scenario.flat_rate_changes:
                flat_rate = scenario.flat_rate_changes[firm.sic]
            
            vat_liability = turnover * flat_rate
            return vat_liability, 'flat_rate'
        
        # Standard VAT calculation
        vat_rate = scenario.standard_rate
        
        # Check for reduced rate
        if scenario.reduced_rate and scenario.split_rate_target:
            if self._qualifies_for_reduced_rate(firm, scenario.split_rate_target):
                vat_rate = scenario.reduced_rate
        
        # Output VAT (only on B2C)
        b2c_share = 1 - firm.b2b_share
        output_vat = turnover * b2c_share * vat_rate
        
        # Input VAT credit
        costs = turnover * (1 - firm.gross_margin)
        input_vat_credit = costs * firm.input_vat_share * scenario.standard_rate
        
        net_vat = max(0, output_vat - input_vat_credit)
        
        # Apply taper if applicable
        if scenario.threshold_design == 'taper':
            if scenario.T_start <= turnover < scenario.T_full:
                taper_factor = (turnover - scenario.T_start) / (scenario.T_full - scenario.T_start)
                net_vat *= taper_factor
        
        return net_vat, 'standard'
    
    def _qualifies_for_reduced_rate(self, firm: EnhancedFirm, target: Dict) -> bool:
        """Check if firm qualifies for reduced rate"""
        
        if 'sic_list' in target:
            return firm.sic in target['sic_list']
        
        if 'high_employment' in target:
            return firm.employment >= target['high_employment']
        
        return False
    
    def simulate_with_compliance(self, firm: EnhancedFirm, 
                                scenario: PolicyScenarioEnhanced) -> Dict:
        """Simulate firm with compliance costs and enforcement"""
        
        # Determine threshold for registration decision
        if scenario.threshold_design == 'notch':
            threshold = scenario.T
        else:
            threshold = scenario.T_start
        
        # Initial registration decision (before bunching)
        would_register = firm.turnover_baseline >= threshold
        
        # Calculate compliance cost
        compliance_cost = firm.get_compliance_cost(would_register)
        
        # Bunching decision considering compliance costs
        if would_register:
            # Calculate benefit of staying below threshold
            vat_if_registered, scheme = self.calculate_vat_liability_enhanced(
                firm, scenario, firm.turnover_baseline, True
            )
            
            total_burden = vat_if_registered + compliance_cost
            
            # Enhanced bunching with compliance consideration
            if firm.turnover_baseline < threshold + 20:  # Near threshold
                # Probability of bunching increases with burden
                bunching_incentive = total_burden / firm.turnover_baseline
                
                # Adjust bunching elasticity
                adjusted_bunching = firm.bunching_elasticity * (1 + bunching_incentive)
                
                # Calculate bunching adjustment
                distance_from_threshold = firm.turnover_baseline - threshold
                if distance_from_threshold > 0 and distance_from_threshold < 10:
                    
                    # Consider audit risk
                    audit_prob = ENFORCEMENT_PARAMETERS['audit_probability']['at_threshold']
                    audit_risk = audit_prob * scenario.enforcement_intensity
                    
                    # Reduce bunching if high audit risk
                    adjusted_bunching *= (1 - audit_risk * 2)
                    
                    # Calculate turnover adjustment
                    max_adjustment = distance_from_threshold + 1
                    adjustment = min(max_adjustment, adjusted_bunching * 10)
                    
                    turnover_reported = max(threshold - 1, firm.turnover_baseline - adjustment)
                else:
                    turnover_reported = firm.turnover_baseline
            else:
                turnover_reported = firm.turnover_baseline
        else:
            turnover_reported = firm.turnover_baseline
        
        # Final registration status
        registered = turnover_reported >= threshold
        
        # Calculate final VAT liability
        vat_liability, scheme_used = self.calculate_vat_liability_enhanced(
            firm, scenario, turnover_reported, registered
        )
        
        # Calculate profit impact
        revenue_change = (turnover_reported - firm.turnover_baseline) * firm.gross_margin
        
        if registered:
            total_cost = vat_liability + compliance_cost
        else:
            total_cost = 0
        
        profit_change = revenue_change - total_cost
        
        # Cash flow impact (if registered)
        if registered:
            working_capital_days = CASH_FLOW_PARAMETERS['payment_terms']['vat_payment_days']
            cash_flow_cost = (vat_liability * working_capital_days / 365 * 
                            CASH_FLOW_PARAMETERS['working_capital_cost'])
        else:
            cash_flow_cost = 0
        
        return {
            'firm_id': firm.firm_id,
            'sic': firm.sic,
            'employment': firm.employment,
            'turnover_baseline': firm.turnover_baseline,
            'turnover_reported': turnover_reported,
            'registered': registered,
            'vat_liability': vat_liability,
            'compliance_cost': compliance_cost if registered else 0,
            'scheme_used': scheme_used,
            'profit_change': profit_change,
            'cash_flow_cost': cash_flow_cost,
            'effective_rate': vat_liability / turnover_reported if turnover_reported > 0 else 0,
            'total_burden': vat_liability + compliance_cost + cash_flow_cost if registered else 0,
            'weight': firm.weight
        }


# ============================================================================
# ANALYSIS AND VISUALIZATION
# ============================================================================

def plot_enhanced_comparison(results_df: pd.DataFrame):
    """Create comprehensive comparison plots"""
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    
    # 1. Turnover distribution with bunching
    ax = axes[0, 0]
    for scenario in results_df['scenario_id'].unique():
        scenario_df = results_df[results_df['scenario_id'] == scenario]
        weights = scenario_df['weight'].values
        turnovers = scenario_df['turnover_reported'].values
        
        # Create weighted histogram
        hist, bins = np.histogram(turnovers, bins=30, weights=weights)
        bin_centers = (bins[:-1] + bins[1:]) / 2
        ax.plot(bin_centers, hist, label=scenario, alpha=0.7)
    
    ax.axvline(x=90, color='red', linestyle='--', alpha=0.5)
    ax.set_xlabel('Turnover (£k)')
    ax.set_ylabel('Weighted Firm Count')
    ax.set_title('Turnover Distribution by Scenario')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 2. Total burden (VAT + compliance) by turnover
    ax = axes[0, 1]
    for scenario in ['S0', 'S2_enhanced']:
        if scenario in results_df['scenario_id'].unique():
            scenario_df = results_df[results_df['scenario_id'] == scenario]
            
            # Group by turnover bins
            scenario_df['turnover_bin'] = pd.cut(scenario_df['turnover_baseline'], 
                                                 bins=range(70, 121, 5))
            burden_by_bin = scenario_df.groupby('turnover_bin')['total_burden'].mean()
            
            ax.plot(range(70, 116, 5), burden_by_bin.values, 
                   label=scenario, marker='o')
    
    ax.set_xlabel('Turnover (£k)')
    ax.set_ylabel('Total Burden (£k)')
    ax.set_title('Total Burden (VAT + Compliance + Cash Flow)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 3. Compliance cost impact
    ax = axes[0, 2]
    compliance_impact = results_df.groupby('scenario_id').agg({
        'compliance_cost': 'mean',
        'vat_liability': 'mean',
        'cash_flow_cost': 'mean'
    })
    
    compliance_impact.plot(kind='bar', stacked=True, ax=ax)
    ax.set_ylabel('Average Cost (£k)')
    ax.set_title('Cost Breakdown by Scenario')
    ax.legend(title='Cost Type')
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)
    
    # 4. Sector-specific impacts
    ax = axes[1, 0]
    sector_impacts = results_df.groupby(['scenario_id', 'sic'])['total_burden'].mean().unstack()
    main_sectors = sector_impacts.sum().nlargest(5).index
    sector_impacts[main_sectors].T.plot(kind='bar', ax=ax)
    ax.set_xlabel('Sector (SIC)')
    ax.set_ylabel('Average Total Burden (£k)')
    ax.set_title('Sector-Specific Burden')
    ax.legend(title='Scenario', bbox_to_anchor=(1.05, 1))
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)
    
    # 5. Registration rates by employment size
    ax = axes[1, 1]
    results_df['size_band'] = pd.cut(results_df['employment'], 
                                     bins=[0, 5, 10, 20, 1000],
                                     labels=['Micro', 'Small', 'Medium', 'Large'])
    
    reg_by_size = results_df.groupby(['scenario_id', 'size_band'])['registered'].mean() * 100
    reg_by_size.unstack().plot(kind='bar', ax=ax)
    ax.set_ylabel('Registration Rate (%)')
    ax.set_title('Registration by Firm Size')
    ax.legend(title='Size')
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)
    
    # 6. Scheme usage
    ax = axes[1, 2]
    scheme_usage = results_df[results_df['registered']].groupby(
        ['scenario_id', 'scheme_used']
    ).size().unstack(fill_value=0)
    
    if not scheme_usage.empty:
        scheme_usage.plot(kind='bar', stacked=True, ax=ax)
        ax.set_ylabel('Number of Firms')
        ax.set_title('VAT Scheme Usage')
        ax.legend(title='Scheme')
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)
    
    plt.suptitle('Enhanced VAT Simulation Results', fontsize=14, y=1.02)
    plt.tight_layout()
    
    return fig


def create_enhanced_scenarios() -> List[PolicyScenarioEnhanced]:
    """Create enhanced policy scenarios"""
    
    return [
        PolicyScenarioEnhanced(
            scenario_id='S0',
            scenario_name='Current System',
            standard_rate=0.20,
            threshold_design='notch',
            T=90.0,
            allow_flat_rate=True,
            enforcement_intensity=1.0
        ),
        PolicyScenarioEnhanced(
            scenario_id='S2_enhanced',
            scenario_name='Taper with Compliance',
            standard_rate=0.20,
            threshold_design='taper',
            T_start=65.0,
            T_full=110.0,
            allow_flat_rate=True,
            enforcement_intensity=1.0
        ),
        PolicyScenarioEnhanced(
            scenario_id='S3_strict',
            scenario_name='Taper with Strict Enforcement',
            standard_rate=0.20,
            threshold_design='taper',
            T_start=90.0,
            T_full=135.0,
            allow_flat_rate=True,
            enforcement_intensity=2.0  # Double audit probability
        ),
        PolicyScenarioEnhanced(
            scenario_id='S4_reformed_flat',
            scenario_name='Reformed Flat Rate',
            standard_rate=0.20,
            threshold_design='notch',
            T=90.0,
            allow_flat_rate=True,
            flat_rate_changes={
                '47': 0.06,  # Lower rate for retail
                '56': 0.10,  # Lower for food service
                '96': 0.11   # Lower for personal services
            },
            enforcement_intensity=1.0
        )
    ]


def main():
    """Run enhanced simulation"""
    
    print("=" * 80)
    print("ENHANCED VAT SIMULATION WITH EMPIRICAL DATA")
    print("=" * 80)
    
    # Initialize simulator with synthetic data
    synthetic_path = Path(__file__).parent.parent / 'analysis' / 'synthetic_firms_turnover.csv'
    
    if not synthetic_path.exists():
        print(f"Error: Synthetic data not found at {synthetic_path}")
        print("Please run generate_synthetic_data.py first")
        return
    
    simulator = EnhancedVATSimulator(str(synthetic_path))
    
    # Create firm sample
    print("\nCreating firm sample from synthetic data...")
    firms = simulator.create_firm_sample(n_firms=5000, turnover_range=(70, 120))
    print(f"Created {len(firms)} firms for simulation")
    
    # Create scenarios
    scenarios = create_enhanced_scenarios()
    
    # Run simulations
    print("\nRunning enhanced simulations...")
    all_results = []
    
    for scenario in scenarios:
        print(f"  Simulating {scenario.scenario_name}...")
        scenario_results = []
        
        for firm in firms:
            result = simulator.simulate_with_compliance(firm, scenario)
            result['scenario_id'] = scenario.scenario_id
            result['scenario_name'] = scenario.scenario_name
            scenario_results.append(result)
        
        all_results.extend(scenario_results)
    
    # Convert to DataFrame
    results_df = pd.DataFrame(all_results)
    
    # Calculate weighted statistics
    print("\n" + "=" * 80)
    print("WEIGHTED STATISTICS BY SCENARIO")
    print("=" * 80)
    
    for scenario_id in results_df['scenario_id'].unique():
        scenario_df = results_df[results_df['scenario_id'] == scenario_id]
        
        # Weight all statistics
        total_weight = scenario_df['weight'].sum()
        
        weighted_stats = {
            'VAT Revenue': (scenario_df['vat_liability'] * scenario_df['weight']).sum() / 1000,  # £m
            'Compliance Costs': (scenario_df['compliance_cost'] * scenario_df['weight']).sum() / 1000,
            'Registration Rate': (scenario_df['registered'] * scenario_df['weight']).sum() / total_weight * 100,
            'Avg Effective Rate': (scenario_df['effective_rate'] * scenario_df['weight']).sum() / total_weight * 100,
            'Bunching Mass': len(scenario_df[(scenario_df['turnover_reported'] >= 85) & 
                                            (scenario_df['turnover_reported'] < 90)])
        }
        
        print(f"\n{scenario_id}: {scenario_df['scenario_name'].iloc[0]}")
        print("-" * 50)
        for stat, value in weighted_stats.items():
            if 'Rate' in stat:
                print(f"  {stat}: {value:.1f}%")
            else:
                print(f"  {stat}: {value:.2f}")
    
    # Sector analysis
    print("\n" + "=" * 80)
    print("SECTOR-SPECIFIC IMPACTS (S0 vs S2_enhanced)")
    print("=" * 80)
    
    s0_df = results_df[results_df['scenario_id'] == 'S0']
    s2_df = results_df[results_df['scenario_id'] == 'S2_enhanced']
    
    sectors = s0_df.groupby('sic').size().nlargest(10).index
    
    print(f"{'Sector':<10} {'S0 Burden':<12} {'S2 Burden':<12} {'Change':<10}")
    print("-" * 44)
    
    for sic in sectors:
        s0_burden = s0_df[s0_df['sic'] == sic]['total_burden'].mean()
        s2_burden = s2_df[s2_df['sic'] == sic]['total_burden'].mean()
        change = (s2_burden - s0_burden) / s0_burden * 100 if s0_burden > 0 else 0
        
        print(f"{sic:<10} £{s0_burden:<11.2f} £{s2_burden:<11.2f} {change:+.1f}%")
    
    # Save results
    output_dir = Path(__file__).parent / 'results'
    output_dir.mkdir(exist_ok=True)
    
    results_df.to_csv(output_dir / 'enhanced_simulation_results.csv', index=False)
    print(f"\nResults saved to {output_dir / 'enhanced_simulation_results.csv'}")
    
    # Create visualizations
    print("\nGenerating visualizations...")
    fig = plot_enhanced_comparison(results_df)
    fig.savefig(output_dir / 'enhanced_vat_analysis.png', dpi=300, bbox_inches='tight')
    print(f"Plots saved to {output_dir / 'enhanced_vat_analysis.png'}")
    
    # Key findings
    print("\n" + "=" * 80)
    print("KEY FINDINGS")
    print("=" * 80)
    
    print("\n1. COMPLIANCE COSTS MATTER:")
    avg_compliance = results_df[results_df['registered']]['compliance_cost'].mean()
    avg_vat = results_df[results_df['registered']]['vat_liability'].mean()
    print(f"   Average compliance cost: £{avg_compliance:.2f}k")
    print(f"   Average VAT liability: £{avg_vat:.2f}k")
    print(f"   Compliance as % of VAT: {avg_compliance/avg_vat*100:.1f}%")
    
    print("\n2. BUNCHING PATTERNS:")
    for scenario_id in ['S0', 'S2_enhanced']:
        if scenario_id in results_df['scenario_id'].unique():
            scenario_df = results_df[results_df['scenario_id'] == scenario_id]
            bunching = len(scenario_df[(scenario_df['turnover_reported'] >= 85) & 
                                      (scenario_df['turnover_reported'] < 90)])
            print(f"   {scenario_id}: {bunching} firms bunching")
    
    print("\n3. FLAT RATE SCHEME USAGE:")
    flat_rate_usage = results_df[
        (results_df['registered']) & 
        (results_df['scheme_used'] == 'flat_rate')
    ].groupby('scenario_id').size()
    for scenario_id, count in flat_rate_usage.items():
        total_reg = results_df[(results_df['scenario_id'] == scenario_id) & 
                              (results_df['registered'])].shape[0]
        print(f"   {scenario_id}: {count}/{total_reg} ({count/total_reg*100:.1f}%)")
    
    print("\n" + "=" * 80)
    print("SIMULATION COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    main()