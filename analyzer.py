"""
Business analysis module for evaluating opportunities.
Includes metrics calculation, SWOT analysis, and scoring.
Uses OpenAI for intelligent analysis.
"""

import os
import json
from dataclasses import dataclass, field
from typing import Optional
from dotenv import load_dotenv
from scraper import BusinessData

load_dotenv()

# Try to import OpenAI
try:
    from openai import OpenAI
    OPENAI_AVAILABLE = bool(os.getenv("OPENAI_API_KEY"))
except ImportError:
    OPENAI_AVAILABLE = False


@dataclass
class BusinessMetrics:
    """Calculated business metrics."""
    roi_percent: Optional[float] = None  # (Cash Flow / Price) * 100
    price_to_revenue: Optional[float] = None  # Price / Revenue
    price_to_cash_flow: Optional[float] = None  # Price / Cash Flow (multiple)
    payback_years: Optional[float] = None  # Price / Annual Cash Flow
    cash_flow_margin: Optional[float] = None  # Cash Flow / Revenue * 100

    def to_dict(self) -> dict:
        return {
            'ROI (%)': f"{self.roi_percent:.1f}%" if self.roi_percent else "N/A",
            'Price/Revenue': f"{self.price_to_revenue:.2f}x" if self.price_to_revenue else "N/A",
            'Price/CF Multiple': f"{self.price_to_cash_flow:.1f}x" if self.price_to_cash_flow else "N/A",
            'Payback (years)': f"{self.payback_years:.1f}" if self.payback_years else "N/A",
            'CF Margin (%)': f"{self.cash_flow_margin:.1f}%" if self.cash_flow_margin else "N/A",
        }


@dataclass
class SWOTAnalysis:
    """SWOT analysis results."""
    strengths: list = field(default_factory=list)
    weaknesses: list = field(default_factory=list)
    opportunities: list = field(default_factory=list)
    threats: list = field(default_factory=list)

    def summary(self) -> str:
        """Return a short summary string."""
        return f"S:{len(self.strengths)} W:{len(self.weaknesses)} O:{len(self.opportunities)} T:{len(self.threats)}"

    def to_dict(self) -> dict:
        return {
            'Strengths': self.strengths,
            'Weaknesses': self.weaknesses,
            'Opportunities': self.opportunities,
            'Threats': self.threats,
        }


@dataclass
class ViabilityAssessment:
    """AI-powered viability assessment."""
    is_viable: bool = False
    confidence: str = "Low"  # Low, Medium, High
    viability_score: int = 0  # 0-100
    key_risks: list = field(default_factory=list)
    key_opportunities: list = field(default_factory=list)
    red_flags: list = field(default_factory=list)
    due_diligence_items: list = field(default_factory=list)
    verdict: str = ""


@dataclass
class AnalysisResult:
    """Complete analysis result for a business."""
    business: BusinessData
    metrics: BusinessMetrics
    swot: SWOTAnalysis
    overall_score: float = 0.0
    recommendation: str = ""
    rank: int = 0
    viability: Optional[ViabilityAssessment] = None
    ai_enhanced: bool = False


class BusinessAnalyzer:
    """Analyzer for evaluating business opportunities."""

    # Benchmark values for scoring (industry averages)
    GOOD_ROI_THRESHOLD = 25.0  # 25% ROI is good
    EXCELLENT_ROI_THRESHOLD = 40.0  # 40%+ is excellent
    GOOD_PAYBACK_YEARS = 4.0  # 4 years or less is good
    EXCELLENT_PAYBACK_YEARS = 2.5  # 2.5 years or less is excellent

    def calculate_metrics(self, business: BusinessData) -> BusinessMetrics:
        """
        Calculate financial metrics for a business.

        Args:
            business: BusinessData object with scraped information

        Returns:
            BusinessMetrics object with calculated values
        """
        metrics = BusinessMetrics()

        # ROI: (Cash Flow / Asking Price) * 100
        if business.cash_flow and business.asking_price and business.asking_price > 0:
            metrics.roi_percent = (business.cash_flow / business.asking_price) * 100

        # Price to Revenue ratio
        if business.asking_price and business.gross_revenue and business.gross_revenue > 0:
            metrics.price_to_revenue = business.asking_price / business.gross_revenue

        # Price to Cash Flow multiple
        if business.asking_price and business.cash_flow and business.cash_flow > 0:
            metrics.price_to_cash_flow = business.asking_price / business.cash_flow

        # Payback period in years
        if business.asking_price and business.cash_flow and business.cash_flow > 0:
            metrics.payback_years = business.asking_price / business.cash_flow

        # Cash flow margin
        if business.cash_flow and business.gross_revenue and business.gross_revenue > 0:
            metrics.cash_flow_margin = (business.cash_flow / business.gross_revenue) * 100

        return metrics

    def generate_swot(self, business: BusinessData, metrics: BusinessMetrics) -> SWOTAnalysis:
        """
        Generate SWOT analysis based on business data and metrics.

        Args:
            business: BusinessData object
            metrics: Calculated BusinessMetrics

        Returns:
            SWOTAnalysis object
        """
        swot = SWOTAnalysis()

        # === STRENGTHS ===

        # Strong ROI
        if metrics.roi_percent and metrics.roi_percent >= self.GOOD_ROI_THRESHOLD:
            if metrics.roi_percent >= self.EXCELLENT_ROI_THRESHOLD:
                swot.strengths.append(f"Excellent ROI of {metrics.roi_percent:.1f}% - well above market average")
            else:
                swot.strengths.append(f"Strong ROI of {metrics.roi_percent:.1f}%")

        # Quick payback
        if metrics.payback_years and metrics.payback_years <= self.GOOD_PAYBACK_YEARS:
            if metrics.payback_years <= self.EXCELLENT_PAYBACK_YEARS:
                swot.strengths.append(f"Very fast payback period of {metrics.payback_years:.1f} years")
            else:
                swot.strengths.append(f"Reasonable payback period of {metrics.payback_years:.1f} years")

        # Established business
        if business.year_established:
            import datetime
            years_in_business = datetime.datetime.now().year - business.year_established
            if years_in_business >= 10:
                swot.strengths.append(f"Well-established business ({years_in_business} years in operation)")
            elif years_in_business >= 5:
                swot.strengths.append(f"Established business track record ({years_in_business} years)")

        # Strong revenue
        if business.gross_revenue and business.gross_revenue >= 500000:
            swot.strengths.append(f"Solid revenue base (${business.gross_revenue:,.0f}/year)")

        # Employees indicate operational infrastructure
        if business.employees and business.employees >= 3:
            swot.strengths.append(f"Existing team of {business.employees} employees")

        # Good cash flow margin
        if metrics.cash_flow_margin and metrics.cash_flow_margin >= 20:
            swot.strengths.append(f"Healthy profit margins ({metrics.cash_flow_margin:.1f}% cash flow margin)")

        # Positive highlights from listing
        positive_keywords = ['growing', 'loyal', 'repeat', 'established', 'profitable', 'turnkey', 'trained']
        for highlight in business.highlights:
            if any(kw in highlight.lower() for kw in positive_keywords):
                swot.strengths.append(highlight[:100])
                if len(swot.strengths) >= 6:
                    break

        # === WEAKNESSES ===

        # Low ROI
        if metrics.roi_percent and metrics.roi_percent < 15:
            swot.weaknesses.append(f"Below-average ROI of {metrics.roi_percent:.1f}%")
        elif metrics.roi_percent is None and business.asking_price:
            swot.weaknesses.append("Cash flow data not disclosed - unable to calculate ROI")

        # Long payback period
        if metrics.payback_years and metrics.payback_years > 5:
            swot.weaknesses.append(f"Long payback period of {metrics.payback_years:.1f} years")

        # New/unproven business
        if business.year_established:
            import datetime
            years_in_business = datetime.datetime.now().year - business.year_established
            if years_in_business < 3:
                swot.weaknesses.append(f"Relatively new business ({years_in_business} years) - limited track record")

        # High price relative to revenue
        if metrics.price_to_revenue and metrics.price_to_revenue > 1.5:
            swot.weaknesses.append(f"Premium pricing at {metrics.price_to_revenue:.2f}x revenue")

        # Missing key financial data
        if not business.cash_flow:
            swot.weaknesses.append("Cash flow not disclosed - due diligence required")
        if not business.gross_revenue:
            swot.weaknesses.append("Revenue not disclosed - financial verification needed")

        # Low margins
        if metrics.cash_flow_margin and metrics.cash_flow_margin < 10:
            swot.weaknesses.append(f"Thin profit margins ({metrics.cash_flow_margin:.1f}%)")

        # Few or no employees might mean owner-dependent
        if business.employees is not None and business.employees <= 1:
            swot.weaknesses.append("May be owner-operated - transition risk")

        # === OPPORTUNITIES ===

        # Room for margin improvement
        if metrics.cash_flow_margin and metrics.cash_flow_margin < 15:
            swot.opportunities.append("Potential to improve operational efficiency and margins")

        # Small employee base could mean growth potential
        if business.employees and 2 <= business.employees <= 5:
            swot.opportunities.append("Room for team expansion to scale operations")

        # Revenue growth potential
        if business.gross_revenue and business.gross_revenue < 1000000:
            swot.opportunities.append("Growth potential to scale past $1M revenue")

        # Digital/marketing opportunities (common for SMBs)
        swot.opportunities.append("Potential for digital marketing and online presence expansion")

        # Operational improvements
        swot.opportunities.append("Opportunity to implement modern systems and processes")

        # Strategic keywords from description
        opportunity_keywords = ['expand', 'growth', 'potential', 'opportunity', 'untapped', 'scalable']
        for keyword in opportunity_keywords:
            if keyword in business.description.lower():
                swot.opportunities.append(f"Seller notes growth potential in listing")
                break

        # === THREATS ===

        # Market competition (general)
        swot.threats.append("Market competition from established and new entrants")

        # Economic sensitivity
        if business.category:
            luxury_keywords = ['restaurant', 'retail', 'hospitality', 'entertainment', 'travel']
            if any(kw in business.category.lower() for kw in luxury_keywords):
                swot.threats.append("Business category sensitive to economic downturns")

        # Key person dependency
        if business.employees is not None and business.employees <= 2:
            swot.threats.append("Risk of customer loss during ownership transition")

        # Lease/location risks
        if business.lease_info:
            swot.threats.append("Lease terms and potential rent increases")
        else:
            swot.threats.append("Location dependency - lease terms should be verified")

        # Industry disruption
        swot.threats.append("Potential industry disruption from technology or market changes")

        # Reason for selling analysis
        if business.reason_for_selling:
            concerning_keywords = ['health', 'retiring', 'divorce', 'urgent', 'must sell']
            if any(kw in business.reason_for_selling.lower() for kw in concerning_keywords):
                swot.threats.append("Seller motivation may indicate underlying issues")

        return swot

    def calculate_score(self, business: BusinessData, metrics: BusinessMetrics, swot: SWOTAnalysis) -> float:
        """
        Calculate overall opportunity score (0-100).

        Scoring weights:
        - ROI: 30%
        - Payback period: 25%
        - Business maturity: 20%
        - SWOT balance: 15%
        - Data completeness: 10%

        Args:
            business: BusinessData object
            metrics: Calculated metrics
            swot: SWOT analysis

        Returns:
            Score from 0 to 100
        """
        score = 0.0

        # === ROI Score (30 points max) ===
        if metrics.roi_percent:
            if metrics.roi_percent >= 50:
                score += 30
            elif metrics.roi_percent >= 40:
                score += 27
            elif metrics.roi_percent >= 30:
                score += 24
            elif metrics.roi_percent >= 25:
                score += 20
            elif metrics.roi_percent >= 20:
                score += 16
            elif metrics.roi_percent >= 15:
                score += 12
            elif metrics.roi_percent >= 10:
                score += 8
            else:
                score += 4

        # === Payback Period Score (25 points max) ===
        if metrics.payback_years:
            if metrics.payback_years <= 2:
                score += 25
            elif metrics.payback_years <= 3:
                score += 22
            elif metrics.payback_years <= 4:
                score += 18
            elif metrics.payback_years <= 5:
                score += 14
            elif metrics.payback_years <= 7:
                score += 10
            else:
                score += 5

        # === Business Maturity Score (20 points max) ===
        import datetime
        if business.year_established:
            years = datetime.datetime.now().year - business.year_established
            if years >= 15:
                score += 20
            elif years >= 10:
                score += 17
            elif years >= 7:
                score += 14
            elif years >= 5:
                score += 11
            elif years >= 3:
                score += 7
            else:
                score += 3

        # === SWOT Balance Score (15 points max) ===
        strengths_count = len(swot.strengths)
        weaknesses_count = len(swot.weaknesses)
        opportunities_count = len(swot.opportunities)
        threats_count = len(swot.threats)

        # Positive factors
        positive = strengths_count + opportunities_count
        negative = weaknesses_count + threats_count

        if positive > 0 and negative > 0:
            ratio = positive / (positive + negative)
            score += ratio * 15
        elif positive > 0:
            score += 15

        # === Data Completeness Score (10 points max) ===
        data_points = 0
        if business.asking_price:
            data_points += 1
        if business.cash_flow:
            data_points += 1
        if business.gross_revenue:
            data_points += 1
        if business.year_established:
            data_points += 1
        if business.employees is not None:
            data_points += 1
        if business.description and len(business.description) > 100:
            data_points += 1

        score += (data_points / 6) * 10

        return round(min(score, 100), 1)

    def generate_recommendation(self, score: float, metrics: BusinessMetrics, swot: SWOTAnalysis) -> str:
        """
        Generate a text recommendation based on analysis.

        Args:
            score: Overall score
            metrics: Business metrics
            swot: SWOT analysis

        Returns:
            Recommendation text
        """
        if score >= 80:
            rating = "🌟 HIGHLY RECOMMENDED"
            verdict = "This appears to be an excellent opportunity with strong fundamentals."
        elif score >= 65:
            rating = "✅ RECOMMENDED"
            verdict = "A solid opportunity worth serious consideration."
        elif score >= 50:
            rating = "⚠️ PROCEED WITH CAUTION"
            verdict = "Moderate opportunity - thorough due diligence essential."
        elif score >= 35:
            rating = "⚡ HIGH RISK"
            verdict = "Significant concerns identified - extensive verification needed."
        else:
            rating = "❌ NOT RECOMMENDED"
            verdict = "Too many red flags or insufficient data for confident assessment."

        # Build detailed recommendation
        parts = [f"{rating}", verdict]

        # Key highlights
        if metrics.roi_percent and metrics.roi_percent >= 25:
            parts.append(f"Key strength: {metrics.roi_percent:.1f}% ROI.")

        if swot.strengths:
            parts.append(f"Top strength: {swot.strengths[0]}")

        if swot.weaknesses:
            parts.append(f"Main concern: {swot.weaknesses[0]}")

        return " ".join(parts)

    def _build_business_context(self, business: BusinessData, metrics: BusinessMetrics) -> str:
        """Build context string for LLM analysis."""
        price_str = f"${business.asking_price:,.0f}" if business.asking_price else "Not disclosed"
        cashflow_str = f"${business.cash_flow:,.0f}/year" if business.cash_flow else "Not disclosed"
        revenue_str = f"${business.gross_revenue:,.0f}/year" if business.gross_revenue else "Not disclosed"
        roi_str = f"{metrics.roi_percent:.1f}%" if metrics.roi_percent else "N/A"
        payback_str = f"{metrics.payback_years:.1f} years" if metrics.payback_years else "N/A"
        ptr_str = f"{metrics.price_to_revenue:.2f}x" if metrics.price_to_revenue else "N/A"
        cfm_str = f"{metrics.cash_flow_margin:.1f}%" if metrics.cash_flow_margin else "N/A"
        desc_str = business.description[:1500] if business.description else "No description available"
        highlights_str = ", ".join(business.highlights[:5]) if business.highlights else "None listed"

        return f"""
Business Name: {business.name}
Asking Price: {price_str}
Cash Flow: {cashflow_str}
Gross Revenue: {revenue_str}
Location: {business.location or 'Not specified'}
Category: {business.category or 'Not specified'}
Year Established: {business.year_established or 'Not specified'}
Employees: {business.employees or 'Not specified'}

Calculated Metrics:
- ROI: {roi_str}
- Payback Period: {payback_str}
- Price/Revenue Ratio: {ptr_str}
- Cash Flow Margin: {cfm_str}

Business Description:
{desc_str}

Highlights: {highlights_str}
"""

    def _analyze_with_openai(self, business: BusinessData, metrics: BusinessMetrics, rule_based_swot: SWOTAnalysis) -> tuple[SWOTAnalysis, str, ViabilityAssessment]:
        """
        Use OpenAI to generate intelligent SWOT analysis, validate viability,
        and cross-check with rule-based analysis.
        """
        client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        business_context = self._build_business_context(business, metrics)

        # Include rule-based SWOT for cross-checking
        rule_swot_summary = f"""
Rule-Based Analysis Results (for cross-checking):
- Strengths identified: {'; '.join(rule_based_swot.strengths[:3]) if rule_based_swot.strengths else 'None'}
- Weaknesses identified: {'; '.join(rule_based_swot.weaknesses[:3]) if rule_based_swot.weaknesses else 'None'}
- Opportunities identified: {'; '.join(rule_based_swot.opportunities[:3]) if rule_based_swot.opportunities else 'None'}
- Threats identified: {'; '.join(rule_based_swot.threats[:3]) if rule_based_swot.threats else 'None'}
"""

        prompt = f"""You are a senior small business acquisition analyst with 20+ years of experience evaluating business purchases.

TASK: Perform a comprehensive viability analysis of this business acquisition opportunity.

{business_context}

{rule_swot_summary}

Provide a thorough analysis including:

1. **SWOT ANALYSIS** - Cross-check and enhance the rule-based analysis above. Be specific to THIS business.
   - Validate or challenge the automated findings
   - Add insights the automated system may have missed
   - Consider industry-specific factors

2. **VIABILITY ASSESSMENT** - Critical evaluation:
   - Is this business viable for acquisition? (yes/no)
   - Confidence level in your assessment (Low/Medium/High)
   - Viability score (0-100)

3. **RISK ANALYSIS**:
   - Key risks a buyer should be aware of
   - Red flags that warrant immediate attention
   - Critical due diligence items before purchase

4. **RECOMMENDATION** - Clear, actionable advice for a potential buyer

Respond in this exact JSON format:
{{
    "swot": {{
        "strengths": ["specific strength 1", "specific strength 2", ...],
        "weaknesses": ["specific weakness 1", "specific weakness 2", ...],
        "opportunities": ["specific opportunity 1", "specific opportunity 2", ...],
        "threats": ["specific threat 1", "specific threat 2", ...]
    }},
    "viability": {{
        "is_viable": true/false,
        "confidence": "Low/Medium/High",
        "viability_score": 0-100,
        "key_risks": ["risk 1", "risk 2", ...],
        "key_opportunities": ["opportunity 1", "opportunity 2", ...],
        "red_flags": ["red flag 1 if any", ...],
        "due_diligence_items": ["item 1", "item 2", ...],
        "verdict": "2-3 sentence overall verdict on viability"
    }},
    "recommendation": "Your detailed 2-4 sentence recommendation for potential buyer"
}}
"""

        try:
            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {
                        "role": "system",
                        "content": "You are a senior business acquisition analyst. Provide specific, data-driven analysis. Be critical and thorough - buyers depend on your assessment. Respond only with valid JSON."
                    },
                    {"role": "user", "content": prompt}
                ],
                temperature=0.2,
                max_tokens=1500
            )

            # Parse JSON response
            content = response.choices[0].message.content
            # Handle potential markdown code blocks
            if "```json" in content:
                content = content.split("```json")[1].split("```")[0]
            elif "```" in content:
                content = content.split("```")[1].split("```")[0]

            result = json.loads(content.strip())

            # Build SWOT
            swot_data = result.get("swot", {})
            swot = SWOTAnalysis(
                strengths=swot_data.get("strengths", []),
                weaknesses=swot_data.get("weaknesses", []),
                opportunities=swot_data.get("opportunities", []),
                threats=swot_data.get("threats", [])
            )

            # Build Viability Assessment
            viability_data = result.get("viability", {})
            viability = ViabilityAssessment(
                is_viable=viability_data.get("is_viable", False),
                confidence=viability_data.get("confidence", "Low"),
                viability_score=viability_data.get("viability_score", 0),
                key_risks=viability_data.get("key_risks", []),
                key_opportunities=viability_data.get("key_opportunities", []),
                red_flags=viability_data.get("red_flags", []),
                due_diligence_items=viability_data.get("due_diligence_items", []),
                verdict=viability_data.get("verdict", "")
            )

            recommendation = result.get("recommendation", "")

            return swot, recommendation, viability

        except Exception as e:
            print(f"OpenAI analysis failed: {e}, falling back to rule-based")
            return None, None, None

    def analyze(self, business: BusinessData, use_ai: bool = True) -> AnalysisResult:
        """
        Perform complete analysis on a business.

        Args:
            business: BusinessData object
            use_ai: Whether to use OpenAI for analysis (default True)

        Returns:
            AnalysisResult with metrics, SWOT, score, viability, and recommendation
        """
        metrics = self.calculate_metrics(business)

        # Always generate rule-based SWOT first (for cross-checking)
        rule_based_swot = self.generate_swot(business, metrics)

        # Initialize defaults
        swot = rule_based_swot
        recommendation = None
        viability = None
        ai_enhanced = False

        # Try OpenAI analysis if available and enabled
        if use_ai and OPENAI_AVAILABLE:
            ai_swot, ai_recommendation, ai_viability = self._analyze_with_openai(
                business, metrics, rule_based_swot
            )

            if ai_swot is not None:
                # Merge AI SWOT with rule-based (cross-checked)
                swot = self._merge_swot_analyses(rule_based_swot, ai_swot)
                ai_enhanced = True

            if ai_recommendation is not None:
                recommendation = ai_recommendation

            if ai_viability is not None:
                viability = ai_viability

        # Calculate score (uses merged SWOT if AI enhanced)
        score = self.calculate_score(business, metrics, swot)

        # Adjust score based on AI viability assessment if available
        if viability and viability.viability_score > 0:
            # Blend rule-based score with AI viability score (60% rule-based, 40% AI)
            score = round(score * 0.6 + viability.viability_score * 0.4, 1)

        # Generate recommendation if not provided by AI
        if recommendation is None:
            recommendation = self.generate_recommendation(score, metrics, swot)
        else:
            # Prepend score-based rating to AI recommendation
            if score >= 75:
                rating = "🌟 HIGHLY RECOMMENDED"
            elif score >= 60:
                rating = "✅ RECOMMENDED"
            elif score >= 45:
                rating = "⚠️ PROCEED WITH CAUTION"
            elif score >= 30:
                rating = "⚡ HIGH RISK"
            else:
                rating = "❌ NOT RECOMMENDED"
            recommendation = f"{rating} — {recommendation}"

        return AnalysisResult(
            business=business,
            metrics=metrics,
            swot=swot,
            overall_score=score,
            recommendation=recommendation,
            viability=viability,
            ai_enhanced=ai_enhanced
        )

    def _merge_swot_analyses(self, rule_based: SWOTAnalysis, ai_based: SWOTAnalysis) -> SWOTAnalysis:
        """
        Merge rule-based and AI-based SWOT analyses.
        Prioritizes AI insights but includes unique rule-based findings.
        """
        def merge_lists(rule_list: list, ai_list: list, max_items: int = 5) -> list:
            # Start with AI insights (usually more contextual)
            merged = list(ai_list)

            # Add unique rule-based insights not covered by AI
            for item in rule_list:
                # Check if similar insight already exists (simple keyword overlap)
                item_words = set(item.lower().split())
                is_duplicate = False
                for existing in merged:
                    existing_words = set(existing.lower().split())
                    overlap = len(item_words & existing_words)
                    if overlap >= 3:  # If 3+ words overlap, consider it duplicate
                        is_duplicate = True
                        break

                if not is_duplicate and len(merged) < max_items:
                    merged.append(item)

            return merged[:max_items]

        return SWOTAnalysis(
            strengths=merge_lists(rule_based.strengths, ai_based.strengths),
            weaknesses=merge_lists(rule_based.weaknesses, ai_based.weaknesses),
            opportunities=merge_lists(rule_based.opportunities, ai_based.opportunities),
            threats=merge_lists(rule_based.threats, ai_based.threats)
        )


def analyze_businesses(businesses: list[BusinessData], progress_callback=None) -> list[AnalysisResult]:
    """
    Analyze a list of businesses and return ranked results.

    Args:
        businesses: List of BusinessData objects
        progress_callback: Optional callback(current, total, message)

    Returns:
        List of AnalysisResult objects, sorted by score (highest first)
    """
    analyzer = BusinessAnalyzer()
    results = []
    total = len(businesses)

    for i, business in enumerate(businesses):
        if progress_callback:
            progress_callback(i, total, f"Analyzing {business.name[:30]}...")

        try:
            result = analyzer.analyze(business)
            results.append(result)
        except Exception as e:
            # If analysis fails, create a basic result without AI
            print(f"Analysis failed for {business.name}: {e}")
            try:
                result = analyzer.analyze(business, use_ai=False)
                results.append(result)
            except Exception as e2:
                print(f"Fallback analysis also failed: {e2}")
                # Skip this business
                continue

    if progress_callback:
        progress_callback(total, total, "Analysis complete!")

    # Sort by score (descending) and assign ranks
    results.sort(key=lambda x: x.overall_score, reverse=True)

    for i, result in enumerate(results):
        result.rank = i + 1

    return results
