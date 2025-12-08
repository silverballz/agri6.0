"""
ROI and Impact Metrics Calculator

Calculates cost savings, resource efficiency, and return on investment
for precision agriculture using AgriFlux platform.
"""

import numpy as np
from typing import Dict, Optional, Tuple
from dataclasses import dataclass


@dataclass
class FarmParameters:
    """Parameters for ROI calculation"""
    farm_size_ha: float = 100.0
    crop_type: str = "wheat"
    baseline_yield_kg_ha: float = 3000.0
    crop_price_per_kg: float = 0.25
    water_cost_per_m3: float = 0.50
    fertilizer_cost_per_ha: float = 150.0
    pesticide_cost_per_ha: float = 80.0
    labor_cost_per_ha: float = 200.0
    agriflux_annual_cost: float = 5000.0


@dataclass
class ImpactMetrics:
    """Calculated impact metrics"""
    # Cost savings
    yield_improvement_pct: float
    yield_improvement_kg: float
    revenue_increase: float
    
    # Resource efficiency
    water_savings_pct: float
    water_savings_m3: float
    water_cost_savings: float
    
    fertilizer_reduction_pct: float
    fertilizer_cost_savings: float
    
    pesticide_reduction_pct: float
    pesticide_cost_savings: float
    
    # Total savings and ROI
    total_annual_savings: float
    net_benefit: float
    roi_pct: float
    payback_period_years: float
    
    # Environmental impact
    carbon_sequestration_tons: float
    carbon_value: float


class ROICalculator:
    """
    Calculate ROI and impact metrics for precision agriculture.
    
    Based on research and industry benchmarks:
    - Early detection can improve yields by 5-15%
    - Precision irrigation saves 20-40% water
    - Targeted fertilization reduces usage by 15-30%
    - Precision pesticide application reduces usage by 20-40%
    """
    
    # Default improvement factors (conservative estimates)
    DEFAULT_YIELD_IMPROVEMENT = 0.08  # 8% yield improvement
    DEFAULT_WATER_SAVINGS = 0.25  # 25% water savings
    DEFAULT_FERTILIZER_REDUCTION = 0.20  # 20% fertilizer reduction
    DEFAULT_PESTICIDE_REDUCTION = 0.30  # 30% pesticide reduction
    
    # Environmental factors
    CARBON_PRICE_PER_TON = 25.0  # USD per ton CO2
    BIOMASS_TO_CARBON_FACTOR = 0.45  # 45% of biomass is carbon
    
    def __init__(self, params: Optional[FarmParameters] = None):
        """Initialize calculator with farm parameters"""
        self.params = params or FarmParameters()
    
    def calculate_cost_savings(
        self,
        health_index: float = 0.7,
        alert_response_rate: float = 0.9
    ) -> Dict[str, float]:
        """
        Calculate cost savings from early detection and intervention.
        
        Args:
            health_index: Current average NDVI (0-1)
            alert_response_rate: Percentage of alerts acted upon (0-1)
        
        Returns:
            Dictionary with cost savings metrics
        """
        # Yield improvement based on health monitoring and early intervention
        # Higher health index and response rate = better yield protection
        base_improvement = self.DEFAULT_YIELD_IMPROVEMENT
        health_factor = min(health_index / 0.7, 1.0)  # Normalize to 0.7 as baseline
        response_factor = alert_response_rate
        
        yield_improvement_pct = base_improvement * health_factor * response_factor
        
        # Calculate absolute yield improvement
        baseline_total_yield = self.params.baseline_yield_kg_ha * self.params.farm_size_ha
        improved_yield = baseline_total_yield * (1 + yield_improvement_pct)
        yield_improvement_kg = improved_yield - baseline_total_yield
        
        # Revenue increase from improved yield
        revenue_increase = yield_improvement_kg * self.params.crop_price_per_kg
        
        return {
            'yield_improvement_pct': yield_improvement_pct * 100,
            'yield_improvement_kg': yield_improvement_kg,
            'baseline_yield_kg': baseline_total_yield,
            'improved_yield_kg': improved_yield,
            'revenue_increase': revenue_increase,
            'crop_price_per_kg': self.params.crop_price_per_kg
        }
    
    def calculate_resource_efficiency(
        self,
        irrigation_zones_used: bool = True,
        precision_application: bool = True
    ) -> Dict[str, float]:
        """
        Calculate resource efficiency improvements.
        
        Args:
            irrigation_zones_used: Whether precision irrigation zones are used
            precision_application: Whether precision fertilizer/pesticide application is used
        
        Returns:
            Dictionary with resource efficiency metrics
        """
        # Water savings from precision irrigation
        water_savings_pct = self.DEFAULT_WATER_SAVINGS if irrigation_zones_used else 0.0
        
        # Estimate baseline water usage (m3/ha varies by crop, using 5000 m3/ha as average)
        baseline_water_m3_ha = 5000.0
        total_baseline_water = baseline_water_m3_ha * self.params.farm_size_ha
        water_savings_m3 = total_baseline_water * water_savings_pct
        water_cost_savings = water_savings_m3 * self.params.water_cost_per_m3
        
        # Fertilizer reduction from targeted application
        fertilizer_reduction_pct = self.DEFAULT_FERTILIZER_REDUCTION if precision_application else 0.0
        total_fertilizer_cost = self.params.fertilizer_cost_per_ha * self.params.farm_size_ha
        fertilizer_cost_savings = total_fertilizer_cost * fertilizer_reduction_pct
        
        # Pesticide reduction from targeted application
        pesticide_reduction_pct = self.DEFAULT_PESTICIDE_REDUCTION if precision_application else 0.0
        total_pesticide_cost = self.params.pesticide_cost_per_ha * self.params.farm_size_ha
        pesticide_cost_savings = total_pesticide_cost * pesticide_reduction_pct
        
        return {
            'water_savings_pct': water_savings_pct * 100,
            'water_savings_m3': water_savings_m3,
            'baseline_water_m3': total_baseline_water,
            'water_cost_savings': water_cost_savings,
            
            'fertilizer_reduction_pct': fertilizer_reduction_pct * 100,
            'baseline_fertilizer_cost': total_fertilizer_cost,
            'fertilizer_cost_savings': fertilizer_cost_savings,
            
            'pesticide_reduction_pct': pesticide_reduction_pct * 100,
            'baseline_pesticide_cost': total_pesticide_cost,
            'pesticide_cost_savings': pesticide_cost_savings
        }
    
    def calculate_carbon_impact(
        self,
        mean_ndvi: float = 0.7,
        biomass_increase_pct: float = 0.08
    ) -> Dict[str, float]:
        """
        Calculate carbon sequestration and environmental impact.
        
        Args:
            mean_ndvi: Average NDVI value
            biomass_increase_pct: Percentage increase in biomass from improved practices
        
        Returns:
            Dictionary with carbon impact metrics
        """
        # Estimate biomass from NDVI (simplified model)
        # Typical crop biomass: 5-15 tons/ha dry matter
        # NDVI correlation: higher NDVI = higher biomass
        baseline_biomass_tons_ha = 8.0  # Conservative estimate
        ndvi_factor = min(mean_ndvi / 0.7, 1.2)  # Scale based on NDVI
        
        current_biomass_tons_ha = baseline_biomass_tons_ha * ndvi_factor
        total_biomass_tons = current_biomass_tons_ha * self.params.farm_size_ha
        
        # Additional biomass from improved practices
        additional_biomass = total_biomass_tons * biomass_increase_pct
        
        # Carbon sequestration (biomass * carbon content factor)
        carbon_sequestered_tons = additional_biomass * self.BIOMASS_TO_CARBON_FACTOR
        
        # Convert to CO2 equivalent (C to CO2: multiply by 44/12 = 3.67)
        co2_equivalent_tons = carbon_sequestered_tons * 3.67
        
        # Calculate carbon credit value
        carbon_value = co2_equivalent_tons * self.CARBON_PRICE_PER_TON
        
        return {
            'biomass_tons_ha': current_biomass_tons_ha,
            'total_biomass_tons': total_biomass_tons,
            'additional_biomass_tons': additional_biomass,
            'carbon_sequestered_tons': carbon_sequestered_tons,
            'co2_equivalent_tons': co2_equivalent_tons,
            'carbon_price_per_ton': self.CARBON_PRICE_PER_TON,
            'carbon_value': carbon_value
        }
    
    def calculate_full_roi(
        self,
        health_index: float = 0.7,
        alert_response_rate: float = 0.9,
        irrigation_zones_used: bool = True,
        precision_application: bool = True,
        mean_ndvi: float = 0.7
    ) -> ImpactMetrics:
        """
        Calculate complete ROI and impact metrics.
        
        Args:
            health_index: Current average NDVI
            alert_response_rate: Percentage of alerts acted upon
            irrigation_zones_used: Whether precision irrigation is used
            precision_application: Whether precision application is used
            mean_ndvi: Average NDVI for carbon calculation
        
        Returns:
            ImpactMetrics dataclass with all calculated metrics
        """
        # Calculate individual components
        cost_savings = self.calculate_cost_savings(health_index, alert_response_rate)
        resource_efficiency = self.calculate_resource_efficiency(
            irrigation_zones_used, precision_application
        )
        carbon_impact = self.calculate_carbon_impact(mean_ndvi)
        
        # Total annual savings
        total_annual_savings = (
            cost_savings['revenue_increase'] +
            resource_efficiency['water_cost_savings'] +
            resource_efficiency['fertilizer_cost_savings'] +
            resource_efficiency['pesticide_cost_savings'] +
            carbon_impact['carbon_value']
        )
        
        # Net benefit (savings minus AgriFlux cost)
        net_benefit = total_annual_savings - self.params.agriflux_annual_cost
        
        # ROI percentage
        roi_pct = (net_benefit / self.params.agriflux_annual_cost) * 100
        
        # Payback period
        if net_benefit > 0:
            payback_period_years = self.params.agriflux_annual_cost / net_benefit
        else:
            payback_period_years = float('inf')
        
        return ImpactMetrics(
            yield_improvement_pct=cost_savings['yield_improvement_pct'],
            yield_improvement_kg=cost_savings['yield_improvement_kg'],
            revenue_increase=cost_savings['revenue_increase'],
            
            water_savings_pct=resource_efficiency['water_savings_pct'],
            water_savings_m3=resource_efficiency['water_savings_m3'],
            water_cost_savings=resource_efficiency['water_cost_savings'],
            
            fertilizer_reduction_pct=resource_efficiency['fertilizer_reduction_pct'],
            fertilizer_cost_savings=resource_efficiency['fertilizer_cost_savings'],
            
            pesticide_reduction_pct=resource_efficiency['pesticide_reduction_pct'],
            pesticide_cost_savings=resource_efficiency['pesticide_cost_savings'],
            
            total_annual_savings=total_annual_savings,
            net_benefit=net_benefit,
            roi_pct=roi_pct,
            payback_period_years=payback_period_years,
            
            carbon_sequestration_tons=carbon_impact['carbon_sequestered_tons'],
            carbon_value=carbon_impact['carbon_value']
        )
    
    def get_assumptions(self) -> Dict[str, str]:
        """
        Get transparent list of assumptions used in calculations.
        
        Returns:
            Dictionary of assumption descriptions
        """
        return {
            'Yield Improvement': f'{self.DEFAULT_YIELD_IMPROVEMENT*100:.0f}% improvement from early detection and intervention (industry benchmark: 5-15%)',
            'Water Savings': f'{self.DEFAULT_WATER_SAVINGS*100:.0f}% reduction from precision irrigation (industry benchmark: 20-40%)',
            'Fertilizer Reduction': f'{self.DEFAULT_FERTILIZER_REDUCTION*100:.0f}% reduction from targeted application (industry benchmark: 15-30%)',
            'Pesticide Reduction': f'{self.DEFAULT_PESTICIDE_REDUCTION*100:.0f}% reduction from precision application (industry benchmark: 20-40%)',
            'Carbon Price': f'${self.CARBON_PRICE_PER_TON:.2f} per ton CO2 (voluntary carbon market average)',
            'Biomass to Carbon': f'{self.BIOMASS_TO_CARBON_FACTOR*100:.0f}% of dry biomass is carbon',
            'Farm Parameters': f'Based on user inputs or defaults for {self.params.crop_type}',
            'Conservative Estimates': 'All calculations use conservative estimates to provide realistic expectations'
        }


def format_currency(value: float, currency: str = "USD") -> str:
    """Format value as currency"""
    return f"${value:,.2f} {currency}"


def format_percentage(value: float) -> str:
    """Format value as percentage"""
    return f"{value:.1f}%"


def format_quantity(value: float, unit: str) -> str:
    """Format quantity with unit"""
    return f"{value:,.0f} {unit}"
