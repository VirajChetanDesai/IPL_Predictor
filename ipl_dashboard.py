import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import accuracy_score, mean_squared_error
from sklearn.preprocessing import LabelEncoder
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline


# Set page configuration
st.set_page_config(page_title="IPL Data Analysis Dashboard", layout="wide")

# Load datasets
@st.cache_data
def load_data():
    deliveries = pd.read_csv('deliveries.csv')
    matches = pd.read_csv('matches.csv')
    return deliveries, matches

# Load data
deliveries, matches = load_data()

# Create a new column in matches for easier selection
matches['match_label'] = matches['team1'] + " vs " + matches['team2'] + " (" + matches['date'] + ")"

# App Title
st.title("IPL Data Analysis Dashboard")

# Sidebar for navigation
st.sidebar.title("Navigation")
selected_page = st.sidebar.radio("Select a Page", ["Match Overview", "Team Analysis", "Player Analysis", "Over-by-Over Analysis", "Venue Insights"])


if selected_page == "Match Overview":
    st.header("Match Overview")

    # Improved Match Selection
    selected_match_label = st.selectbox("Select Match:", matches['match_label'].unique())
    selected_match_id = matches[matches['match_label'] == selected_match_label]['id'].values[0]

    # Filter matches based on the selected match ID
    match_details = matches[matches['id'] == selected_match_id].iloc[0]

    # Display Match Summary
    st.subheader(f"Details for {match_details['team1']} vs {match_details['team2']} on {match_details['date']}")
    col1, col2 = st.columns(2)
    with col1:
        st.write(f"**Season**: {match_details['season']}")
        st.write(f"**City**: {match_details['city']}")
        st.write(f"**Venue**: {match_details['venue']}")
    with col2:
        st.write(f"**Match Winner**: {match_details['winner']} by {match_details['result_margin']} {match_details['result']}")
        st.write(f"**Player of the Match**: {match_details['player_of_match']}")
        st.write(f"**Toss Winner**: {match_details['toss_winner']} (chose to {match_details['toss_decision']})")

    # Enhanced Analysis: Runs and Wickets Overview
    st.subheader("Runs and Wickets Summary")
    match_deliveries = deliveries[deliveries['match_id'] == selected_match_id]
    innings_summary = match_deliveries.groupby('inning').agg({
        'total_runs': 'sum',
        'is_wicket': 'sum',
        'extras_type': lambda x: (x != 'NA').sum()
    }).reset_index()
    innings_summary.columns = ['Inning', 'Total Runs', 'Wickets', 'Extras']
    st.table(innings_summary)

    # Visualize runs and wickets per over
    st.subheader("Runs and Wickets per Over")
    runs_wickets_per_over = match_deliveries.groupby(['inning', 'over']).agg({
        'total_runs': 'sum',
        'is_wicket': 'sum'
    }).reset_index()

    fig = go.Figure()
    for inning in runs_wickets_per_over['inning'].unique():
        inning_data = runs_wickets_per_over[runs_wickets_per_over['inning'] == inning]
        fig.add_trace(go.Bar(x=inning_data['over'], y=inning_data['total_runs'], name=f'Runs (Inning {inning})', opacity=0.7))
        fig.add_trace(go.Scatter(x=inning_data['over'], y=inning_data['is_wicket'], mode='markers', name=f'Wickets (Inning {inning})', yaxis='y2'))

    fig.update_layout(
        title='Runs Scored and Wickets Taken in Each Over',
        xaxis_title='Over',
        yaxis_title='Runs',
        yaxis2=dict(title='Wickets', overlaying='y', side='right'),
        barmode='group'
    )
    st.plotly_chart(fig)

    # Player Analysis
    st.subheader("Player Performance Analysis")

    # Top Batsmen
    top_batsmen = match_deliveries.groupby('batter')['batsman_runs'].sum().sort_values(ascending=False).head(5)
    st.write("Top 5 Batsmen:")
    st.table(top_batsmen.reset_index().rename(columns={'batter': 'Batsman', 'batsman_runs': 'Runs Scored'}))

    # Top Bowlers
    top_bowlers = match_deliveries.groupby('bowler').agg({
        'is_wicket': 'sum',
        'total_runs': 'sum'
    }).sort_values('is_wicket', ascending=False).head(5)
    top_bowlers['economy'] = top_bowlers['total_runs'] / (match_deliveries.groupby('bowler').size() / 6)
    st.write("Top 5 Bowlers:")
    st.table(top_bowlers.reset_index().rename(columns={'bowler': 'Bowler', 'is_wicket': 'Wickets', 'economy': 'Economy'})[['Bowler', 'Wickets', 'Economy']])

    # Player of the Match Analysis
    potm = match_details['player_of_match']
    st.subheader(f"Player of the Match: {potm}")
    potm_batting = match_deliveries[match_deliveries['batter'] == potm]['batsman_runs'].sum()
    potm_bowling = match_deliveries[match_deliveries['bowler'] == potm]['is_wicket'].sum()
    st.write(f"Runs Scored: {potm_batting}")
    st.write(f"Wickets Taken: {potm_bowling}")

    # ML Model: Predict Win Probability
    st.subheader("Win Probability Prediction")

    # Prepare data for ML model
    X = matches[['team1', 'team2', 'toss_winner', 'toss_decision', 'venue']]
    y = matches['winner']

    # Remove rows with NaN values
    mask = X.notna().all(axis=1) & y.notna()
    X = X[mask]
    y = y[mask]

    # Create a preprocessing pipeline
    preprocessor = ColumnTransformer(
        transformers=[
            ('encoder', OneHotEncoder(handle_unknown='ignore'), ['team1', 'team2', 'toss_winner', 'toss_decision', 'venue'])
        ])

    # Create a pipeline with preprocessor and random forest classifier
    pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('classifier', RandomForestClassifier(n_estimators=100, random_state=42))
    ])

    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Fit the pipeline
    pipeline.fit(X_train, y_train)

    # Make prediction for the selected match
    selected_match_features = match_details[['team1', 'team2', 'toss_winner', 'toss_decision', 'venue']].to_frame().T
    win_prob = pipeline.predict_proba(selected_match_features)[0]

    # Display win probabilities
    st.write(f"Win probability for {match_details['team1']}: {win_prob[0]:.2%}")
    st.write(f"Win probability for {match_details['team2']}: {win_prob[1]:.2%}")

    # Display model performance metrics
    y_pred = pipeline.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    st.write(f"Model Accuracy: {accuracy:.2%}")

    # Feature importance
    feature_names = pipeline.named_steps['preprocessor'].get_feature_names_out()
    feature_importance = pd.DataFrame({
        'feature': feature_names,
        'importance': pipeline.named_steps['classifier'].feature_importances_
    }).sort_values('importance', ascending=False).head(10)
    st.subheader("Top 10 Features Influencing Win Probability")
    fig = px.bar(feature_importance, x='importance', y='feature', orientation='h')
    st.plotly_chart(fig)
    
elif selected_page == "Team Analysis":
    st.header("Team Performance Analysis")

    # Select Team from a dropdown
    teams = sorted(matches['team1'].unique())
    selected_team = st.selectbox("Select Team:", teams)

    # Filter matches based on the selected team
    team_matches = matches[(matches['team1'] == selected_team) | (matches['team2'] == selected_team)]

    # Display team performance
    st.subheader(f"Performance of {selected_team}")
    total_matches = len(team_matches)
    total_wins = len(team_matches[team_matches['winner'] == selected_team])
    win_percentage = (total_wins / total_matches) * 100

    col1, col2, col3 = st.columns(3)
    col1.metric("Total Matches", total_matches)
    col2.metric("Total Wins", total_wins)
    col3.metric("Win Percentage", f"{win_percentage:.2f}%")

    # Season-wise performance
    st.subheader("Season-wise Performance")
    season_performance = team_matches.groupby('season').agg({
        'id': 'count',
        'winner': lambda x: (x == selected_team).sum()
    }).reset_index()
    season_performance.columns = ['Season', 'Matches', 'Wins']
    season_performance['Win Rate'] = season_performance['Wins'] / season_performance['Matches']

    fig = go.Figure()
    fig.add_trace(go.Bar(x=season_performance['Season'], y=season_performance['Matches'], name='Matches'))
    fig.add_trace(go.Bar(x=season_performance['Season'], y=season_performance['Wins'], name='Wins'))
    fig.add_trace(go.Scatter(x=season_performance['Season'], y=season_performance['Win Rate'], mode='lines+markers', name='Win Rate', yaxis='y2'))
    fig.update_layout(
        title=f"{selected_team}'s Performance Across Seasons",
        xaxis_title="Season",
        yaxis_title="Number of Matches/Wins",
        yaxis2=dict(title='Win Rate', overlaying='y', side='right'),
        barmode='group'
    )
    st.plotly_chart(fig)

    # Home vs Away Performance
    st.subheader("Home vs Away Performance")
    team_matches['is_home'] = np.where(team_matches['team1'] == selected_team, 'Home', 'Away')
    home_away_perf = team_matches.groupby('is_home').agg({
        'id': 'count',
        'winner': lambda x: (x == selected_team).sum()
    }).reset_index()
    home_away_perf.columns = ['Venue', 'Matches', 'Wins']
    home_away_perf['Win Rate'] = home_away_perf['Wins'] / home_away_perf['Matches']
    
    fig = px.bar(home_away_perf, x='Venue', y=['Matches', 'Wins'], barmode='group', title=f"{selected_team}'s Home vs Away Performance")
    fig.add_trace(go.Scatter(x=home_away_perf['Venue'], y=home_away_perf['Win Rate'], mode='lines+markers', name='Win Rate', yaxis='y2'))
    fig.update_layout(yaxis2=dict(title='Win Rate', overlaying='y', side='right'))
    st.plotly_chart(fig)

    # Top Performers
    st.subheader("Top Performers")
    
    # Top Batsmen
    team_deliveries = deliveries[deliveries['batting_team'] == selected_team]
    top_batsmen = team_deliveries.groupby('batter')['batsman_runs'].sum().sort_values(ascending=False).head(5)
    
    # Top Bowlers
    team_bowling = deliveries[deliveries['bowling_team'] == selected_team]
    top_bowlers = team_bowling.groupby('bowler')['is_wicket'].sum().sort_values(ascending=False).head(5)

    col1, col2 = st.columns(2)
    with col1:
        st.write("Top 5 Batsmen")
        st.table(top_batsmen.reset_index().rename(columns={'batter': 'Batsman', 'batsman_runs': 'Runs'}))
    with col2:
        st.write("Top 5 Bowlers")
        st.table(top_bowlers.reset_index().rename(columns={'bowler': 'Bowler', 'is_wicket': 'Wickets'}))

    # Player Form Analysis
    st.subheader("Player Form Analysis")
    
    # Get top 5 batsmen and bowlers
    top_players = list(top_batsmen.index) + list(top_bowlers.index)
    selected_player = st.selectbox("Select Player for Form Analysis:", top_players)

    # Analyze player's performance over last 10 matches
    player_perf = deliveries[(deliveries['batter'] == selected_player) | (deliveries['bowler'] == selected_player)]
    player_perf = player_perf.merge(matches[['id', 'date']], left_on='match_id', right_on='id')
    player_perf = player_perf.sort_values('date', ascending=False)

    last_10_matches = player_perf['match_id'].unique()[:10]
    last_10_perf = player_perf[player_perf['match_id'].isin(last_10_matches)]

    if selected_player in top_batsmen.index:
        perf_metric = last_10_perf.groupby('match_id')['batsman_runs'].sum()
        metric_name = 'Runs Scored'
    else:
        perf_metric = last_10_perf.groupby('match_id')['is_wicket'].sum()
        metric_name = 'Wickets Taken'

    fig = px.line(x=range(1, 11), y=perf_metric.values[::-1], title=f"{selected_player}'s Form in Last 10 Matches")
    fig.update_layout(xaxis_title="Matches Ago", yaxis_title=metric_name)
    st.plotly_chart(fig)

    # Opposition Analysis
    st.subheader("Opposition Analysis")
    
    team_matches['opponent'] = np.where(team_matches['team1'] == selected_team, team_matches['team2'], team_matches['team1'])
    opposition_perf = team_matches.groupby('opponent').agg({
        'id': 'count',
        'winner': lambda x: (x == selected_team).sum()
    }).reset_index()
    opposition_perf.columns = ['Opponent', 'Matches', 'Wins']
    opposition_perf['Win Rate'] = opposition_perf['Wins'] / opposition_perf['Matches']
    opposition_perf = opposition_perf.sort_values('Win Rate', ascending=False)

    fig = px.bar(opposition_perf, x='Opponent', y='Win Rate', 
                 title=f"{selected_team}'s Performance Against Different Teams")
    st.plotly_chart(fig)

    # Strategy Analysis
    st.subheader("Strategy Analysis")

    # Batting First vs Chasing
    team_matches['is_batting_first'] = np.where(
        ((team_matches['team1'] == selected_team) & (team_matches['toss_decision'] == 'bat')) |
        ((team_matches['team2'] == selected_team) & (team_matches['toss_decision'] == 'field')),
        'Batting First', 'Chasing'
    )
    strategy_perf = team_matches.groupby('is_batting_first').agg({
        'id': 'count',
        'winner': lambda x: (x == selected_team).sum()
    }).reset_index()
    strategy_perf.columns = ['Strategy', 'Matches', 'Wins']
    strategy_perf['Win Rate'] = strategy_perf['Wins'] / strategy_perf['Matches']

    fig = px.bar(strategy_perf, x='Strategy', y=['Matches', 'Wins'], barmode='group',
                 title=f"{selected_team}'s Performance: Batting First vs Chasing")
    fig.add_trace(go.Scatter(x=strategy_perf['Strategy'], y=strategy_perf['Win Rate'], mode='lines+markers', name='Win Rate', yaxis='y2'))
    fig.update_layout(yaxis2=dict(title='Win Rate', overlaying='y', side='right'))
    st.plotly_chart(fig)

    # Machine Learning: Predict Match Outcome
    st.subheader("Match Outcome Prediction")

    # Prepare data for ML model
    X = matches[['team1', 'team2', 'toss_winner', 'toss_decision', 'venue']]
    y = (matches['winner'] == selected_team).astype(int)

    # Remove rows with NaN values
    mask = X.notna().all(axis=1) & y.notna()
    X = X[mask]
    y = y[mask]

    # Create a preprocessing pipeline
    preprocessor = ColumnTransformer(
        transformers=[
            ('encoder', OneHotEncoder(handle_unknown='ignore'), ['team1', 'team2', 'toss_winner', 'toss_decision', 'venue'])
        ])

    # Create a pipeline with preprocessor and random forest classifier
    pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('classifier', RandomForestClassifier(n_estimators=100, random_state=42))
    ])

    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Fit the pipeline
    pipeline.fit(X_train, y_train)

    # Make predictions and evaluate
    y_pred = pipeline.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)

    st.write(f"Model Accuracy: {accuracy:.2f}")

    # Interactive prediction
    st.write("Predict match outcome:")
    col1, col2 = st.columns(2)
    with col1:
        opponent = st.selectbox("Opponent Team", [team for team in teams if team != selected_team])
        toss_winner = st.selectbox("Toss Winner", [selected_team, opponent])
    with col2:
        toss_decision = st.selectbox("Toss Decision", ['bat', 'field'])
        venue = st.selectbox("Venue", matches['venue'].unique())

    # Prepare input for prediction
    input_data = pd.DataFrame([[selected_team, opponent, toss_winner, toss_decision, venue]], 
                              columns=['team1', 'team2', 'toss_winner', 'toss_decision', 'venue'])

    # Make prediction
    win_probability = pipeline.predict_proba(input_data)[0][1]
    st.write(f"Probability of {selected_team} winning: {win_probability:.2f}")

    # Feature importance
    feature_names = pipeline.named_steps['preprocessor'].get_feature_names_out()
    feature_importance = pd.DataFrame({
        'feature': feature_names,
        'importance': pipeline.named_steps['classifier'].feature_importances_
    }).sort_values('importance', ascending=False).head(10)

    st.subheader("Factors Influencing Match Outcome")
    fig = px.bar(feature_importance, x='importance', y='feature', orientation='h', 
                 title="Feature Importance for Match Outcome Prediction")
    st.plotly_chart(fig)

# Player Analysis Page
elif selected_page == "Player Analysis":
    st.header("Player Analysis")

    # Player selection
    all_players = sorted(deliveries['batter'].unique())
    selected_player = st.selectbox("Select Player:", all_players)

    # Player overview
    st.subheader(f"Overview for {selected_player}")
    
    player_matches = deliveries[deliveries['batter'] == selected_player]['match_id'].nunique()
    total_runs = deliveries[deliveries['batter'] == selected_player]['batsman_runs'].sum()
    avg_runs = total_runs / player_matches if player_matches > 0 else 0

    col1, col2, col3 = st.columns(3)
    col1.metric("Matches Played", player_matches)
    col2.metric("Total Runs", total_runs)
    col3.metric("Average Runs per Match", f"{avg_runs:.2f}")

    # Performance trend
    st.subheader("Performance Trend")
    player_perf = deliveries[deliveries['batter'] == selected_player].groupby('match_id')['batsman_runs'].sum().reset_index()
    player_perf = player_perf.sort_values('match_id')
    
    fig = px.line(player_perf, x=player_perf.index, y='batsman_runs', title=f"{selected_player}'s Run Scoring Trend")
    fig.update_layout(xaxis_title="Match Number", yaxis_title="Runs Scored")
    st.plotly_chart(fig)

    # Performance against different teams
    st.subheader("Performance Against Different Teams")
    team_performance = deliveries[deliveries['batter'] == selected_player].groupby('bowling_team')['batsman_runs'].agg(['sum', 'count', 'mean']).reset_index()
    team_performance.columns = ['Team', 'Total Runs', 'Balls Faced', 'Average']
    team_performance['Strike Rate'] = (team_performance['Total Runs'] / team_performance['Balls Faced']) * 100
    team_performance = team_performance.sort_values('Total Runs', ascending=False)

    st.table(team_performance)

    # Batting position analysis
    st.subheader("Batting Position Analysis")
    position_performance = deliveries[deliveries['batter'] == selected_player].groupby('inning')['batsman_runs'].agg(['sum', 'count', 'mean']).reset_index()
    position_performance.columns = ['Position', 'Total Runs', 'Balls Faced', 'Average']
    position_performance['Strike Rate'] = (position_performance['Total Runs'] / position_performance['Balls Faced']) * 100

    fig = px.bar(position_performance, x='Position', y='Total Runs', title=f"{selected_player}'s Performance by Batting Position")
    fig.update_layout(xaxis_title="Batting Position", yaxis_title="Total Runs")
    st.plotly_chart(fig)

# Over-by-Over Analysis Page
elif selected_page == "Over-by-Over Analysis":
    st.header("Over-by-Over Analysis")

    # Match selection
    selected_match_label = st.selectbox("Select Match:", matches['match_label'].unique())
    selected_match_id = matches[matches['match_label'] == selected_match_label]['id'].values[0]

    # Filter deliveries for the selected match
    match_deliveries = deliveries[deliveries['match_id'] == selected_match_id]

    # Over-by-over runs
    st.subheader("Runs Scored in Each Over")
    runs_per_over = match_deliveries.groupby(['inning', 'over'])['total_runs'].sum().reset_index()

    fig = px.line(runs_per_over, x='over', y='total_runs', color='inning', title='Runs Scored in Each Over')
    fig.update_layout(xaxis_title="Over", yaxis_title="Runs Scored")
    st.plotly_chart(fig)

    # Wickets per over
    st.subheader("Wickets Taken in Each Over")
    wickets_per_over = match_deliveries.groupby(['inning', 'over'])['is_wicket'].sum().reset_index()

    fig = px.bar(wickets_per_over, x='over', y='is_wicket', color='inning', title='Wickets Taken in Each Over')
    fig.update_layout(xaxis_title="Over", yaxis_title="Wickets Taken")
    st.plotly_chart(fig)

    # Run rate analysis
    st.subheader("Run Rate Analysis")
    runs_per_over['cumulative_runs'] = runs_per_over.groupby('inning')['total_runs'].cumsum()
    runs_per_over['run_rate'] = runs_per_over['cumulative_runs'] / runs_per_over['over']

    fig = px.line(runs_per_over, x='over', y='run_rate', color='inning', title='Run Rate Progression')
    fig.update_layout(xaxis_title="Over", yaxis_title="Run Rate")
    st.plotly_chart(fig)

    # Partnership analysis
    st.subheader("Partnership Analysis")
    partnerships = match_deliveries.groupby(['inning', 'batter', 'non_striker'])['total_runs'].sum().reset_index()
    top_partnerships = partnerships.sort_values('total_runs', ascending=False).head(5)

    st.table(top_partnerships)

# Venue Insights Page
elif selected_page == "Venue Insights":
    st.header("Venue Insights")

    # Define teams
    teams = sorted(matches['team1'].unique())

    # Venue selection
    venues = sorted(matches['venue'].unique())
    selected_venue = st.selectbox("Select Venue:", venues)

    # Filter matches for the selected venue
    venue_matches = matches[matches['venue'] == selected_venue]

    # Venue statistics
    st.subheader(f"Statistics for {selected_venue}")
    total_matches = len(venue_matches)
    avg_first_innings_score = venue_matches['target_runs'].mean() - 1  # Subtracting 1 to get first innings score

    col1, col2 = st.columns(2)
    col1.metric("Total Matches Played", total_matches)
    col2.metric("Average First Innings Score", f"{avg_first_innings_score:.2f}")

    # Winning percentage by batting first vs chasing
    batting_first_wins = venue_matches[venue_matches['toss_decision'] == 'bat']['toss_winner'] == venue_matches[venue_matches['toss_decision'] == 'bat']['winner']
    chasing_wins = venue_matches[venue_matches['toss_decision'] == 'field']['toss_winner'] == venue_matches[venue_matches['toss_decision'] == 'field']['winner']

    win_percentages = pd.DataFrame({
        'Decision': ['Batting First', 'Chasing'],
        'Win Percentage': [batting_first_wins.mean() * 100, chasing_wins.mean() * 100]
    })

    fig = px.bar(win_percentages, x='Decision', y='Win Percentage', title='Win Percentage by Match Strategy')
    st.plotly_chart(fig)

    # Top performers at the venue
    st.subheader("Top Performers at the Venue")

    venue_deliveries = deliveries[deliveries['match_id'].isin(venue_matches['id'])]

    # Top batsmen
    top_batsmen = venue_deliveries.groupby('batter')['batsman_runs'].sum().sort_values(ascending=False).head(5)
    
    # Top bowlers
    top_bowlers = venue_deliveries.groupby('bowler')['is_wicket'].sum().sort_values(ascending=False).head(5)

    col1, col2 = st.columns(2)
    with col1:
        st.write("Top 5 Batsmen")
        st.table(top_batsmen.reset_index().rename(columns={'batter': 'Batsman', 'batsman_runs': 'Runs'}))
    with col2:
        st.write("Top 5 Bowlers")
        st.table(top_bowlers.reset_index().rename(columns={'bowler': 'Bowler', 'is_wicket': 'Wickets'}))

    # Pitch behavior analysis
    st.subheader("Pitch Behavior Analysis")

    # Average runs per over
    avg_runs_per_over = venue_deliveries.groupby('over')['total_runs'].mean().reset_index()
    fig = px.line(avg_runs_per_over, x='over', y='total_runs', title='Average Runs per Over')
    fig.update_layout(xaxis_title="Over", yaxis_title="Average Runs")
    st.plotly_chart(fig)

    # Average wickets per over
    avg_wickets_per_over = venue_deliveries.groupby('over')['is_wicket'].mean().reset_index()
    fig = px.line(avg_wickets_per_over, x='over', y='is_wicket', title='Average Wickets per Over')
    fig.update_layout(xaxis_title="Over", yaxis_title="Average Wickets")
    st.plotly_chart(fig)

    # Boundary percentage
    total_balls = len(venue_deliveries)
    boundary_balls = len(venue_deliveries[venue_deliveries['batsman_runs'].isin([4, 6])])
    boundary_percentage = (boundary_balls / total_balls) * 100
    st.write(f"Boundary Percentage: {boundary_percentage:.2f}%")

    # Average first innings score vs second innings score
    first_innings_scores = venue_matches['target_runs'] - 1  # Subtracting 1 to get first innings score
    second_innings_scores = venue_matches[venue_matches['result'] != 'tie']['result_margin']
    
    avg_first_innings = first_innings_scores.mean()
    avg_second_innings = second_innings_scores.mean()

    st.write(f"Average First Innings Score: {avg_first_innings:.2f}")
    st.write(f"Average Second Innings Score: {avg_second_innings:.2f}")

    # Toss decision analysis
    st.subheader("Toss Decision Analysis")
    toss_decisions = venue_matches['toss_decision'].value_counts()
    fig = px.pie(values=toss_decisions.values, names=toss_decisions.index, title='Toss Decisions')
    st.plotly_chart(fig)

    # Win percentage after winning toss
    toss_winners = venue_matches[venue_matches['toss_winner'] == venue_matches['winner']]
    toss_win_percentage = (len(toss_winners) / len(venue_matches)) * 100
    st.write(f"Win Percentage After Winning Toss: {toss_win_percentage:.2f}%")

    # Team performance at this venue
    st.subheader("Team Performance at this Venue")
    team_performance = venue_matches.groupby('winner').size().sort_values(ascending=False).reset_index()
    team_performance.columns = ['Team', 'Wins']
    fig = px.bar(team_performance, x='Team', y='Wins', title='Team Wins at this Venue')
    st.plotly_chart(fig)

    # Highest and lowest scores
    highest_score = venue_matches['target_runs'].max() - 1
    lowest_score = venue_matches['target_runs'].min() - 1
    st.write(f"Highest Score: {highest_score}")
    st.write(f"Lowest Score: {lowest_score}")

    # Machine Learning: Predict First Innings Score
    st.subheader("First Innings Score Prediction")

    # Prepare data for ML model
    X = venue_matches[['team1', 'team2', 'toss_winner', 'toss_decision']]
    y = venue_matches['target_runs'] - 1  # Subtracting 1 to get first innings score

    # Remove rows with NaN values
    mask = X.notna().all(axis=1) & y.notna()
    X = X[mask]
    y = y[mask]

    # Create a preprocessing pipeline
    preprocessor = ColumnTransformer(
        transformers=[
            ('encoder', OneHotEncoder(handle_unknown='ignore'), ['team1', 'team2', 'toss_winner', 'toss_decision'])
        ])

    # Create a pipeline with preprocessor and random forest regressor
    pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('regressor', RandomForestRegressor(n_estimators=100, random_state=42))
    ])

    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Fit the pipeline
    pipeline.fit(X_train, y_train)

    # Model performance
    y_pred = pipeline.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    st.write(f"Model Performance (Root Mean Squared Error): {rmse:.2f} runs")

    # Interactive prediction
    st.write("Predict first innings score:")
    col1, col2 = st.columns(2)
    with col1:
        team1 = st.selectbox("Team 1", teams)
        team2 = st.selectbox("Team 2", [team for team in teams if team != team1])
    with col2:
        toss_winner = st.selectbox("Toss Winner", [team1, team2])
        toss_decision = st.selectbox("Toss Decision", ['bat', 'field'])

    # Prepare input for prediction
    input_data = pd.DataFrame([[team1, team2, toss_winner, toss_decision]], 
                              columns=['team1', 'team2', 'toss_winner', 'toss_decision'])

    # Make prediction
    predicted_score = pipeline.predict(input_data)[0]
    st.write(f"Predicted first innings score: {predicted_score:.0f}")

    # Feature importance
    feature_names = pipeline.named_steps['preprocessor'].get_feature_names_out()
    feature_importance = pd.DataFrame({
        'feature': feature_names,
        'importance': pipeline.named_steps['regressor'].feature_importances_
    }).sort_values('importance', ascending=False).head(10)

    st.subheader("Factors Influencing First Innings Score")
    fig = px.bar(feature_importance, x='importance', y='feature', orientation='h', 
                 title="Feature Importance for First Innings Score Prediction")
    st.plotly_chart(fig)

# Footer
st.sidebar.markdown("---")
st.sidebar.markdown("Created by Viraj Desai")
st.sidebar.markdown("Data source: [IPL Dataset](https://www.kaggle.com/datasets/patrickb1912/ipl-complete-dataset-20082020)")