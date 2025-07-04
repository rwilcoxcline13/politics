"""
Municipal and Federal Election Voter Analysis and Propensity Scoring

KEY FEATURES:
- Calculates municipal and federal election participation metrics (2012-2024)
- Categorizes voters into propensity groups: Never Voter, Lost Voter, Sporadic Voter, Consistent Voter, Super Voter
- Generates activation priority scores (1-5) for campaign targeting
- Creates school committee likelihood scores based on demographics and voting history
- Produces comprehensive analysis reports and specialized output files
- Handles both municipal (annual) and federal (biennial) election cycles

INPUT REQUIREMENTS:
- CSV file with voter data containing columns like:
  * vb.vf_mYYYY (municipal voting history: Y/N for each year)
  * vb.vf_gYYYY (federal voting history: Y/N for each year)
  * vb.voterbase_age, vb.voterbase_marital_status (demographics)
  * ts.tsmart_local_voter_score (local voting propensity)
  * ts.tsmart_children_present_score (family composition)
  * enh.tsmart_enhanced_hh_size (household size)
  * gsyn.synth_hh_pct_less_than_35_age (household age composition)

OUTPUT FILES:
- Main file: {input}_with_election_metrics.csv (all voters with new metrics)
- Never voted municipal: {input}_never_voted_municipal.csv
- Never voted federal: {input}_never_voted_federal.csv
- Never voted any: {input}_never_voted_any_election.csv
- High priority municipal targets: {input}_high_priority_municipal_targets.csv
- High priority federal targets: {input}_high_priority_federal_targets.csv
- Federal-only voters: {input}_federal_only_voters.csv

USAGE:
    python municipal_propensity.py
    # Processes 'quincy_voters_final_with_all_donors_improved.csv' by default
"""

import pandas as pd
import numpy as np
from datetime import datetime

def calculate_municipal_and_federal_metrics(df):
    """
    Calculate municipal and federal election metrics for voter data.
    
    Parameters:
    df (DataFrame): Voter data with columns like vb.vf_m2021, vb.vf_g2020, etc.
    
    Returns:
    DataFrame: Original data with new municipal and federal metrics columns
    """
    
    # Create a copy to avoid modifying original
    df_metrics = df.copy()
    
    # Get current year
    current_year = 2025
    
    # Municipal election year columns
    municipal_years = [f'vb.vf_m{year}' for year in range(2024, 2011, -1)]
    
    # Federal election columns (general elections)
    federal_years = [f'vb.vf_g{year}' for year in range(2024, 2011, -1)]
    
    # Initialize new columns - Municipal
    df_metrics['voted_municipal_2021'] = 0
    df_metrics['municipal_elections_count'] = 0
    df_metrics['municipal_years_voted'] = ''
    df_metrics['most_recent_municipal_year'] = np.nan
    df_metrics['years_since_last_municipal'] = np.nan
    df_metrics['municipal_propensity_category'] = ''
    df_metrics['school_committee_likelihood_score'] = 0
    df_metrics['municipal_activation_priority'] = 3
    df_metrics['never_voted_municipal'] = 0
    
    # Initialize new columns - Federal
    df_metrics['voted_federal_2024'] = 0
    df_metrics['voted_federal_2022'] = 0
    df_metrics['voted_federal_2020'] = 0
    df_metrics['federal_elections_count'] = 0
    df_metrics['federal_years_voted'] = ''
    df_metrics['most_recent_federal_year'] = np.nan
    df_metrics['years_since_last_federal'] = np.nan
    df_metrics['federal_propensity_category'] = ''
    df_metrics['federal_activation_priority'] = 3
    df_metrics['never_voted_federal'] = 0
    df_metrics['federal_voter_consistency_score'] = 0
    
    for idx, row in df_metrics.iterrows():
        # === MUNICIPAL METRICS ===
        # 1. Check if voted in 2021 municipal
        if 'vb.vf_m2021' in row and row['vb.vf_m2021'] == 'Y':
            df_metrics.at[idx, 'voted_municipal_2021'] = 1
        
        # 2. Count municipal elections and track years
        municipal_count = 0
        years_voted = []
        most_recent = None
        
        for year in range(2024, 2011, -1):
            col = f'vb.vf_m{year}'
            if col in row and row[col] == 'Y':
                municipal_count += 1
                years_voted.append(str(year))
                if most_recent is None:
                    most_recent = year
        
        df_metrics.at[idx, 'municipal_elections_count'] = municipal_count
        df_metrics.at[idx, 'municipal_years_voted'] = ','.join(years_voted)
        
        # 3. Calculate years since last municipal
        if most_recent:
            df_metrics.at[idx, 'most_recent_municipal_year'] = most_recent
            df_metrics.at[idx, 'years_since_last_municipal'] = current_year - most_recent
        else:
            df_metrics.at[idx, 'never_voted_municipal'] = 1
        
        # 4. Determine municipal propensity category
        if municipal_count == 0:
            muni_category = "Never Voter"
        elif most_recent and (current_year - most_recent) > 4:
            muni_category = "Lost Voter"
        elif municipal_count <= 2 or (most_recent and (current_year - most_recent) > 2):
            muni_category = "Sporadic Voter"
        else:
            # Check for Super Voter status
            try:
                reg_date = pd.to_datetime(row.get('vb.vf_registration_date', '2012'))
                reg_year = reg_date.year
                years_registered = current_year - reg_year
                eligible_elections = min(years_registered, 13)
                participation_rate = municipal_count / eligible_elections if eligible_elections > 0 else 0
                
                if participation_rate >= 0.75:
                    muni_category = "Super Voter"
                else:
                    muni_category = "Consistent Voter"
            except:
                muni_category = "Consistent Voter"
        
        df_metrics.at[idx, 'municipal_propensity_category'] = muni_category
        
        # === FEDERAL METRICS ===
        # 1. Check recent federal elections
        if 'vb.vf_g2024' in row and row['vb.vf_g2024'] == 'Y':
            df_metrics.at[idx, 'voted_federal_2024'] = 1
        if 'vb.vf_g2022' in row and row['vb.vf_g2022'] == 'Y':
            df_metrics.at[idx, 'voted_federal_2022'] = 1
        if 'vb.vf_g2020' in row and row['vb.vf_g2020'] == 'Y':
            df_metrics.at[idx, 'voted_federal_2020'] = 1
        
        # 2. Count federal elections and track years
        federal_count = 0
        federal_years_voted = []
        most_recent_federal = None
        
        for year in range(2024, 2011, -1):
            col = f'vb.vf_g{year}'
            if col in row and row[col] == 'Y':
                federal_count += 1
                federal_years_voted.append(str(year))
                if most_recent_federal is None:
                    most_recent_federal = year
        
        df_metrics.at[idx, 'federal_elections_count'] = federal_count
        df_metrics.at[idx, 'federal_years_voted'] = ','.join(federal_years_voted)
        
        # 3. Calculate years since last federal
        if most_recent_federal:
            df_metrics.at[idx, 'most_recent_federal_year'] = most_recent_federal
            df_metrics.at[idx, 'years_since_last_federal'] = current_year - most_recent_federal
        else:
            df_metrics.at[idx, 'never_voted_federal'] = 1
        
        # 4. Determine federal propensity category
        if federal_count == 0:
            fed_category = "Never Voter"
        elif most_recent_federal and (current_year - most_recent_federal) > 6:
            fed_category = "Lost Voter"
        elif federal_count <= 2 or (most_recent_federal and (current_year - most_recent_federal) > 4):
            fed_category = "Sporadic Voter"
        else:
            # Check for Super Voter status in federal elections
            try:
                reg_date = pd.to_datetime(row.get('vb.vf_registration_date', '2012'))
                reg_year = reg_date.year
                years_registered = current_year - reg_year
                # Federal elections occur every 2 years
                eligible_fed_elections = min(years_registered // 2, 7)  # 2012-2024
                fed_participation_rate = federal_count / eligible_fed_elections if eligible_fed_elections > 0 else 0
                
                if fed_participation_rate >= 0.75:
                    fed_category = "Super Voter"
                else:
                    fed_category = "Consistent Voter"
            except:
                fed_category = "Consistent Voter"
        
        df_metrics.at[idx, 'federal_propensity_category'] = fed_category
        
        # 5. Federal voter consistency score (0-100)
        # Based on voting in presidential (2020, 2016, 2012) and midterm (2022, 2018, 2014) cycles
        consistency_score = 0
        presidential_years = [2020, 2016, 2012]
        midterm_years = [2022, 2018, 2014]
        
        pres_voted = sum(1 for year in presidential_years if row.get(f'vb.vf_g{year}') == 'Y')
        midterm_voted = sum(1 for year in midterm_years if row.get(f'vb.vf_g{year}') == 'Y')
        
        # Presidential consistency (50 points max)
        consistency_score += (pres_voted / len(presidential_years)) * 50
        # Midterm consistency (50 points max)
        consistency_score += (midterm_voted / len(midterm_years)) * 50
        
        df_metrics.at[idx, 'federal_voter_consistency_score'] = int(round(consistency_score))
        
        # 5. Calculate school committee likelihood score
        score = 0
        
        # Municipal participation (40 points max)
        score += min(municipal_count * 10, 40)
        
        # Local voter score (20 points max)
        local_score = row.get('ts.tsmart_local_voter_score', 0)
        if pd.notna(local_score):
            score += (float(local_score) / 100) * 20
        
        # Demographics (20 points max)
        age = row.get('vb.voterbase_age', 0)
        if pd.notna(age) and 28 <= age <= 55:
            score += 10
        
        if row.get('vb.voterbase_marital_status') == 'Married':
            score += 5
        
        children_score = row.get('ts.tsmart_children_present_score', 0)
        if pd.notna(children_score) and children_score > 50:
            score += 5
        
        # Household composition (20 points max)
        hh_size = row.get('enh.tsmart_enhanced_hh_size', 1)
        if pd.notna(hh_size) and 3 <= hh_size <= 5:
            score += 10
        
        hh_young_pct = row.get('gsyn.synth_hh_pct_less_than_35_age', 0)
        if pd.notna(hh_young_pct) and 20 < hh_young_pct < 60:
            score += 10
        
        df_metrics.at[idx, 'school_committee_likelihood_score'] = int(round(score))
        
        # 6. Municipal activation priority
        age = row.get('vb.voterbase_age', 0)
        if muni_category == "Super Voter":
            muni_priority = 1
        elif muni_category == "Never Voter" and pd.notna(age) and age > 65:
            muni_priority = 2
        elif muni_category == "Sporadic Voter" and most_recent and (current_year - most_recent) <= 4:
            muni_priority = 5
        elif muni_category == "Lost Voter" and municipal_count >= 2:
            muni_priority = 4
        elif pd.notna(age) and age <= 35 and municipal_count >= 1:
            muni_priority = 4
        else:
            muni_priority = 3
        
        df_metrics.at[idx, 'municipal_activation_priority'] = muni_priority
        
        # 7. Federal activation priority
        if fed_category == "Super Voter":
            fed_priority = 1
        elif fed_category == "Never Voter" and pd.notna(age) and age > 65:
            fed_priority = 2
        elif fed_category == "Sporadic Voter" and most_recent_federal and (current_year - most_recent_federal) <= 2:
            fed_priority = 5
        elif fed_category == "Lost Voter" and federal_count >= 2:
            fed_priority = 4
        elif pd.notna(age) and age <= 35 and federal_count >= 1:
            fed_priority = 4
        else:
            fed_priority = 3
        
        df_metrics.at[idx, 'federal_activation_priority'] = fed_priority
    
    return df_metrics

def analyze_municipal_and_federal_metrics(df):
    """
    Generate summary statistics for municipal and federal metrics.
    """
    print("MUNICIPAL AND FEDERAL ELECTION METRICS ANALYSIS")
    print("=" * 60)
    
    # Overall statistics
    total_voters = len(df)
    
    # MUNICIPAL SECTION
    print("\n" + "="*60)
    print("MUNICIPAL ELECTIONS ANALYSIS")
    print("="*60)
    
    never_voted_muni = df['never_voted_municipal'].sum()
    voted_2021 = df['voted_municipal_2021'].sum()
    
    print(f"\nTotal Voters Analyzed: {total_voters:,}")
    print(f"Never Voted in Municipal Elections: {never_voted_muni:,} ({never_voted_muni/total_voters*100:.1f}%)")
    print(f"Voted in 2021 Municipal Election: {voted_2021:,} ({voted_2021/total_voters*100:.1f}%)")
    
    # Municipal category breakdown
    print("\nMUNICIPAL VOTER CATEGORIES:")
    print("-" * 30)
    muni_category_counts = df['municipal_propensity_category'].value_counts()
    for category, count in muni_category_counts.items():
        print(f"{category}: {count:,} ({count/total_voters*100:.1f}%)")
    
    # FEDERAL SECTION
    print("\n" + "="*60)
    print("FEDERAL ELECTIONS ANALYSIS")
    print("="*60)
    
    never_voted_fed = df['never_voted_federal'].sum()
    voted_2024 = df['voted_federal_2024'].sum()
    voted_2022 = df['voted_federal_2022'].sum()
    voted_2020 = df['voted_federal_2020'].sum()
    
    print(f"\nTotal Voters Analyzed: {total_voters:,}")
    print(f"Never Voted in Federal Elections: {never_voted_fed:,} ({never_voted_fed/total_voters*100:.1f}%)")
    print(f"Voted in 2024 Federal Election: {voted_2024:,} ({voted_2024/total_voters*100:.1f}%)")
    print(f"Voted in 2022 Federal Election: {voted_2022:,} ({voted_2022/total_voters*100:.1f}%)")
    print(f"Voted in 2020 Federal Election: {voted_2020:,} ({voted_2020/total_voters*100:.1f}%)")
    
    # Federal category breakdown
    print("\nFEDERAL VOTER CATEGORIES:")
    print("-" * 30)
    fed_category_counts = df['federal_propensity_category'].value_counts()
    for category, count in fed_category_counts.items():
        print(f"{category}: {count:,} ({count/total_voters*100:.1f}%)")
    
    # Federal consistency score
    print("\nFEDERAL VOTER CONSISTENCY:")
    print("-" * 30)
    print(f"Average Consistency Score: {df['federal_voter_consistency_score'].mean():.1f}/100")
    print(f"Highly Consistent (80-100): {len(df[df['federal_voter_consistency_score'] >= 80]):,} voters")
    print(f"Moderately Consistent (50-79): {len(df[(df['federal_voter_consistency_score'] >= 50) & (df['federal_voter_consistency_score'] < 80)]):,} voters")
    print(f"Low Consistency (0-49): {len(df[df['federal_voter_consistency_score'] < 50]):,} voters")
    
    # COMPARATIVE ANALYSIS
    print("\n" + "="*60)
    print("COMPARATIVE ANALYSIS: MUNICIPAL VS FEDERAL")
    print("="*60)
    
    # Cross-tabulation of voting behavior
    print("\nVOTER OVERLAP:")
    both_never = len(df[(df['never_voted_municipal'] == 1) & (df['never_voted_federal'] == 1)])
    muni_only_never = len(df[(df['never_voted_municipal'] == 1) & (df['never_voted_federal'] == 0)])
    fed_only_never = len(df[(df['never_voted_municipal'] == 0) & (df['never_voted_federal'] == 1)])
    both_vote = len(df[(df['never_voted_municipal'] == 0) & (df['never_voted_federal'] == 0)])
    
    print(f"Never voted in either: {both_never:,} ({both_never/total_voters*100:.1f}%)")
    print(f"Never voted municipal only: {muni_only_never:,} ({muni_only_never/total_voters*100:.1f}%)")
    print(f"Never voted federal only: {fed_only_never:,} ({fed_only_never/total_voters*100:.1f}%)")
    print(f"Voted in both types: {both_vote:,} ({both_vote/total_voters*100:.1f}%)")
    
    # Drop-off analysis
    print("\nDROP-OFF ANALYSIS:")
    federal_voters = df[df['never_voted_federal'] == 0]
    if len(federal_voters) > 0:
        muni_dropoff = len(federal_voters[federal_voters['never_voted_municipal'] == 1])
        print(f"Federal voters who never vote municipal: {muni_dropoff:,} ({muni_dropoff/len(federal_voters)*100:.1f}%)")
    
    # Activation priority comparison
    print("\nACTIVATION PRIORITY COMPARISON:")
    print("-" * 30)
    print("High Priority (4-5) Targets:")
    high_pri_muni = len(df[df['municipal_activation_priority'] >= 4])
    high_pri_fed = len(df[df['federal_activation_priority'] >= 4])
    high_pri_both = len(df[(df['municipal_activation_priority'] >= 4) & (df['federal_activation_priority'] >= 4)])
    
    print(f"  Municipal: {high_pri_muni:,} voters")
    print(f"  Federal: {high_pri_fed:,} voters")
    print(f"  Both: {high_pri_both:,} voters")
    
    # School committee likelihood for different segments
    print("\n\nSCHOOL COMMITTEE LIKELIHOOD BY SEGMENT:")
    print("-" * 40)
    segments = [
        ("Municipal Super Voters", df[df['municipal_propensity_category'] == 'Super Voter']),
        ("Municipal Never Voters", df[df['never_voted_municipal'] == 1]),
        ("Federal Only Voters", df[(df['never_voted_federal'] == 0) & (df['never_voted_municipal'] == 1)]),
        ("Young Voters (≤35)", df[df['vb.voterbase_age'] <= 35])
    ]
    
    for segment_name, segment_df in segments:
        if len(segment_df) > 0:
            avg_score = segment_df['school_committee_likelihood_score'].mean()
            print(f"{segment_name}: {avg_score:.1f}/100 (n={len(segment_df):,})")
    
    return df

# Main execution function
def process_voter_file(filename):
    """
    Main function to process voter file and add municipal and federal metrics.
    """
    print(f"Loading {filename}...")
    
    # Read CSV with low_memory=False to handle mixed types
    df = pd.read_csv(filename, low_memory=False)
    print(f"Loaded {len(df):,} voter records")
    
    # Calculate metrics
    print("\nCalculating municipal and federal election metrics...")
    df_with_metrics = calculate_municipal_and_federal_metrics(df)
    
    # Analyze and display results
    analyze_municipal_and_federal_metrics(df_with_metrics)
    
    # Save results
    output_filename = filename.replace('.csv', '_with_election_metrics.csv')
    df_with_metrics.to_csv(output_filename, index=False)
    print(f"\n\nResults saved to: {output_filename}")
    
    # Create specialized output files
    
    # 1. Never voted in municipal elections
    never_voters_muni = df_with_metrics[df_with_metrics['never_voted_municipal'] == 1]
    never_muni_filename = filename.replace('.csv', '_never_voted_municipal.csv')
    never_voters_muni.to_csv(never_muni_filename, index=False)
    print(f"Municipal never voters saved to: {never_muni_filename} ({len(never_voters_muni):,} records)")
    
    # 2. Never voted in federal elections
    never_voters_fed = df_with_metrics[df_with_metrics['never_voted_federal'] == 1]
    never_fed_filename = filename.replace('.csv', '_never_voted_federal.csv')
    never_voters_fed.to_csv(never_fed_filename, index=False)
    print(f"Federal never voters saved to: {never_fed_filename} ({len(never_voters_fed):,} records)")
    
    # 3. Never voted in either
    never_both = df_with_metrics[(df_with_metrics['never_voted_municipal'] == 1) & 
                                 (df_with_metrics['never_voted_federal'] == 1)]
    never_both_filename = filename.replace('.csv', '_never_voted_any_election.csv')
    never_both.to_csv(never_both_filename, index=False)
    print(f"Never voted in any election saved to: {never_both_filename} ({len(never_both):,} records)")
    
    # 4. High-priority municipal targets
    high_priority_muni = df_with_metrics[df_with_metrics['municipal_activation_priority'] >= 4]
    priority_muni_filename = filename.replace('.csv', '_high_priority_municipal_targets.csv')
    high_priority_muni.to_csv(priority_muni_filename, index=False)
    print(f"High priority municipal targets saved to: {priority_muni_filename} ({len(high_priority_muni):,} records)")
    
    # 5. High-priority federal targets
    high_priority_fed = df_with_metrics[df_with_metrics['federal_activation_priority'] >= 4]
    priority_fed_filename = filename.replace('.csv', '_high_priority_federal_targets.csv')
    high_priority_fed.to_csv(priority_fed_filename, index=False)
    print(f"High priority federal targets saved to: {priority_fed_filename} ({len(high_priority_fed):,} records)")
    
    # 6. Federal-only voters (vote federal but not municipal)
    federal_only = df_with_metrics[(df_with_metrics['never_voted_municipal'] == 1) & 
                                   (df_with_metrics['never_voted_federal'] == 0)]
    federal_only_filename = filename.replace('.csv', '_federal_only_voters.csv')
    federal_only.to_csv(federal_only_filename, index=False)
    print(f"Federal-only voters saved to: {federal_only_filename} ({len(federal_only):,} records)")
    
    return df_with_metrics

# Example usage
if __name__ == "__main__":
    # Process the Quincy voters file
    df_processed = process_voter_file('quincy_voters_final_with_all_donors_improved.csv')
    
    # Optional: Get specific insights
    print("\n\nADDITIONAL INSIGHTS:")
    print("-" * 50)
    
    # Municipal insights
    print("\nMUNICIPAL ELECTION INSIGHTS:")
    super_voters_muni = df_processed[df_processed['municipal_propensity_category'] == 'Super Voter']
    print(f"Municipal Super Voters (potential champions): {len(super_voters_muni):,}")
    
    young_never_muni = df_processed[
        (df_processed['never_voted_municipal'] == 1) & 
        (df_processed['vb.voterbase_age'] <= 35)
    ]
    print(f"Young Municipal Never Voters (age ≤ 35): {len(young_never_muni):,}")
    
    high_school_never = df_processed[
        (df_processed['never_voted_municipal'] == 1) & 
        (df_processed['school_committee_likelihood_score'] >= 50)
    ]
    print(f"Never Municipal Voters with High School Likelihood (≥50): {len(high_school_never):,}")
    
    # Federal insights
    print("\nFEDERAL ELECTION INSIGHTS:")
    super_voters_fed = df_processed[df_processed['federal_propensity_category'] == 'Super Voter']
    print(f"Federal Super Voters: {len(super_voters_fed):,}")
    
    young_never_fed = df_processed[
        (df_processed['never_voted_federal'] == 1) & 
        (df_processed['vb.voterbase_age'] <= 35)
    ]
    print(f"Young Federal Never Voters (age ≤ 35): {len(young_never_fed):,}")
    
    # Drop-off opportunity
    federal_only_parents = df_processed[
        (df_processed['never_voted_municipal'] == 1) & 
        (df_processed['never_voted_federal'] == 0) &
        (df_processed['school_committee_likelihood_score'] >= 40)
    ]
    print(f"\nFederal-only voters with school-age likelihood (≥40): {len(federal_only_parents):,}")
    print("(These are prime targets for municipal activation)")
    
    # Consistency insights
    highly_consistent = df_processed[df_processed['federal_voter_consistency_score'] >= 80]
    print(f"\nHighly consistent federal voters (≥80 score): {len(highly_consistent):,}")
    
    inconsistent_but_recent = df_processed[
        (df_processed['federal_voter_consistency_score'] < 50) &
        (df_processed['voted_federal_2024'] == 1)
    ]
    print(f"Inconsistent but voted in 2024: {len(inconsistent_but_recent):,}")