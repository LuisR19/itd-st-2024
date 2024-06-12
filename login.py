import json
import pandas as pd
import streamlit as st
from pymongo import MongoClient
from bson.objectid import ObjectId
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt

def connect_to_mongo():
    client = MongoClient("mongodb+srv://aldoparada:AldoParada0805@cluster0.tstzpba.mongodb.net/?retryWrites=true&w=majority&appName=Cluster0")
    db = client["nba"]
    return db

def fetch_data(db, collection_name, columns):
    collection = db[collection_name]
    data = list(collection.find({}, {col: 1 for col in columns}))
    return pd.DataFrame(data)

def fetch_user(db, username, password):
    collection = db["users"]
    user = collection.find_one({"username": username, "password": password})
    return user

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

        # Create plots using Streamlit's base plotting functions
        if 'PTS' in actual_cols:
            st.bar_chart(nba['PTS'])
            top_scorers = nba[['Player', 'PTS']].sort_values(by='PTS', ascending=False).head(10)
            st.write("Top 10 Scorers")
            st.bar_chart(top_scorers.set_index('Player'))

        if 'AST' in actual_cols and 'PTS' in actual_cols:
            st.line_chart(nba[['AST', 'PTS']])
            top_assists = nba[['Player', 'AST']].sort_values(by='AST', ascending=False).head(10)
            st.write("Top 10 Assist Leaders")
            st.bar_chart(top_assists.set_index('Player'))

        if 'TRB' in actual_cols:
            st.area_chart(nba['TRB'])
            top_rebounds = nba[['Player', 'TRB']].sort_values(by='TRB', ascending=False).head(10)
            st.write("Top 10 Rebound Leaders")
            st.bar_chart(top_rebounds.set_index('Player'))

        if 'STL' in actual_cols:
            top_steals = nba[['Player', 'STL']].sort_values(by='STL', ascending=False).head(10)
            st.write("Top 10 Steal Leaders")
            st.bar_chart(top_steals.set_index('Player'))

        if 'BLK' in actual_cols:
            top_blocks = nba[['Player', 'BLK']].sort_values(by='BLK', ascending=False).head(10)
            st.write("Top 10 Block Leaders")
            st.bar_chart(top_blocks.set_index('Player'))

        # Train a simple scikit-learn model
        if 'PTS' in actual_cols and 'AST' in actual_cols and 'TRB' in actual_cols:
            st.write("## Scikit-Learn Model")

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

            # Plot actual vs predicted points using Streamlit's plotting functions
            results = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred})
            st.write("### Actual vs Predicted Points")
            st.line_chart(results)

            # Show the regression line
            fig, ax = plt.subplots()
            ax.scatter(X_test['AST'], y_test, color='blue', label='Actual Points')
            ax.scatter(X_test['AST'], y_pred, color='red', label='Predicted Points')
            ax.set_xlabel('Assists')
            ax.set_ylabel('Points')
            ax.legend()
            st.pyplot(fig)
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
