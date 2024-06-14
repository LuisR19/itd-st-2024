import json
import pandas as pd
import streamlit as st
from pymongo import MongoClient
from bson.objectid import ObjectId
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import plotly.express as px
import plotly.graph_objects as go

def connect_to_mongo():
    client = MongoClient("")
    db = client["nba"]
    return db

def fetch_data(db, collection_name, columns):
    collection = db[collection_name]
    data = list(collection.find({}, {col: 1 for col in columns}))
    return pd.DataFrame(data)

def fetch_user(db, username, password):
    collection = db["users"]
    return collection.find_one({"username": username, "password": password})

def fetch_roles(db):
    collection = db["roles"]
    return list(collection.find({}))

def fetch_data_dictionary(db):
    collection = db["dds"]
    return list(collection.find({}))

def get_column_reference(column_ref, data_dicts):
    file_key, col_idx = column_ref.split('.')
    col_idx = int(col_idx) - 1
    data_dict = data_dicts[file_key]
    return data_dict[col_idx]['nickname']

def log_user_activity(username, user_role, cols):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    log_entry = {
        "timestamp": timestamp,
        "username": username,
        "role": user_role,
        "columns_accessed": cols
    }

    try:
        with open('logs.json', 'a') as log_file:
            json.dump(log_entry, log_file)
            log_file.write('\n')
    except Exception as e:
        st.error(f"Error logging user activity: {e}")

def login_section():
    st.title("Login")

    username = st.text_input("Enter your username")
    password = st.text_input("Enter your password", type="password")

    if st.button("Login"):
        db = connect_to_mongo()
        user = fetch_user(db, username, password)

        if user:
            st.session_state.logged_in = True
            st.session_state.username = username
            st.session_state.role = user['role']
            st.experimental_rerun()
        else:
            st.error("Invalid username or password")

def home_section():
    st.title("NBA Data Viewer")

    db = connect_to_mongo()
    roles = fetch_roles(db)
    data_dicts = {
        'DD': fetch_data_dictionary(db)
    }

    username = st.session_state.username
    user_role = st.session_state.role
    role_info = next((role for role in roles if role['role'] == user_role), None)

    if role_info:
        cols = role_info['cols']
        actual_cols = [get_column_reference(col, data_dicts) for col in cols]

        nba = fetch_data(db, "players", actual_cols)

        # Log user activity
        log_user_activity(username, user_role, actual_cols)

        # Create a search input for a specific player
        search_player = st.text_input("Search for a specific player")
        if st.button("Submit Search"):
            if search_player:
                nba = nba[nba['Player'].str.contains(search_player, case=False, na=False)]

        st.dataframe(nba)

        for col in cols:
            col_info = get_column_reference(col, data_dicts)
            col_details = next((item for item in data_dicts['DD'] if item["nickname"] == col_info), None)
            if col_details:
                st.write(f"Column: {col_info}, Description: {col_details['golden_name']}, Type: {col_details['data_type']}, Access: {col_details['data_access']}")

        # Create plots using Plotly
        if 'PTS' in actual_cols:
            st.write("### Top 10 Scorers")
            top_scorers = nba[['Player', 'PTS']].sort_values(by='PTS', ascending=False).head(10)
            fig = px.bar(top_scorers, x='Player', y='PTS', title="Top 10 Scorers", color='PTS', labels={'PTS': 'Points'}, color_continuous_scale='Blues')
            st.plotly_chart(fig)

        if 'AST' in actual_cols and 'PTS' in actual_cols:
            st.write("### Top 10 Assist Leaders")
            top_assists = nba[['Player', 'AST']].sort_values(by='AST', ascending=False).head(10)
            fig = px.bar(top_assists, x='Player', y='AST', title="Top 10 Assist Leaders", color='AST', labels={'AST': 'Assists'}, color_continuous_scale='Greens')
            st.plotly_chart(fig)

        if 'TRB' in actual_cols:
            st.write("### Top 10 Rebound Leaders")
            top_rebounds = nba[['Player', 'TRB']].sort_values(by='TRB', ascending=False).head(10)
            fig = px.bar(top_rebounds, x='Player', y='TRB', title="Top 10 Rebound Leaders", color='TRB', labels={'TRB': 'Rebounds'}, color_continuous_scale='Oranges')
            st.plotly_chart(fig)

        if 'STL' in actual_cols:
            st.write("### Top 10 Steal Leaders")
            top_steals = nba[['Player', 'STL']].sort_values(by='STL', ascending=False).head(10)
            fig = px.bar(top_steals, x='Player', y='STL', title="Top 10 Steal Leaders", color='STL', labels={'STL': 'Steals'}, color_continuous_scale='Purples')
            st.plotly_chart(fig)

        if 'BLK' in actual_cols:
            st.write("### Top 10 Block Leaders")
            top_blocks = nba[['Player', 'BLK']].sort_values(by='BLK', ascending=False).head(10)
            fig = px.bar(top_blocks, x='Player', y='BLK', title="Top 10 Block Leaders", color='BLK', labels={'BLK': 'Blocks'}, color_continuous_scale='Reds')
            st.plotly_chart(fig)

        # Train a simple scikit-learn model
        if 'PTS' in actual_cols and 'AST' in actual_cols and 'TRB' in actual_cols:
            st.write("## NBA Model")

            # Prepare the data
            X = nba[['AST', 'TRB']]
            y = nba['PTS']

            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

            # Train the model
            model = LinearRegression()
            model.fit(X_train, y_train)

            # Make predictions
            y_pred = model.predict(X_test)

            # Calculate and display the performance metrics
            mse = mean_squared_error(y_test, y_pred)
            st.write(f"Mean Squared Error: {mse}")

            # Create a DataFrame with actual, predicted values and player names
            results = pd.DataFrame({'AST': X_test['AST'], 'TRB': X_test['TRB'], 'Actual': y_test, 'Predicted': y_pred})
            results['Player'] = nba.loc[results.index, 'Player'].values

            # Display the DataFrame with actual and predicted points
            st.write("### Actual and Predicted Points")
            st.dataframe(results)

            # Plot actual vs predicted points using Plotly with player names and trend line
            st.write("### Actual vs Predicted Points")
            fig = px.scatter(results, x='Actual', y='Predicted', hover_data=['Player'], labels={'Actual': 'Actual Points', 'Predicted': 'Predicted Points'}, title='Actual vs Predicted Points', color='Predicted', color_continuous_scale='Viridis', trendline="ols")
            st.plotly_chart(fig)

            # Show the regression line with player names
            st.write("### Assists vs Points with Predicted Points")
            fig = px.scatter(results, x='AST', y='Actual', hover_data=['Player'], labels={'AST': 'Assists', 'Actual': 'Points'}, title='Assists vs Points')
            fig.add_scatter(x=results['AST'], y=results['Predicted'], mode='markers', name='Predicted Points', marker=dict(color='red'))
            st.plotly_chart(fig)
    else:
        st.error("Role not found.")

def main():
    if 'logged_in' not in st.session_state:
        st.session_state.logged_in = False

    if st.session_state.logged_in:
        home_section()
    else:
        login_section()

if __name__ == "__main__":
    main()
