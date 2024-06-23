import streamlit as st
import pandas as pd
import pickle
from sklearn.preprocessing import StandardScaler

model = pickle.load(open('FINAL_XGBRegressor.pkl', 'rb'))
training_data = pd.read_csv('unscaled_ratings.csv')
scaler = StandardScaler()
scaler.fit(training_data[['potential', 'value_eur', 'wage_eur', 'movement_reactions', 'mentality_composure']])

def main():
    st.title("FIFA Player Rating Predictor")
    
    # Input fields for 5 features
    potential = st.number_input("Potential: ", min_value=0, max_value = 99, value=0)
    value_eur = st.number_input("Value (Euros): ", min_value=0, max_value = 1000000000, value=0)
    wage_eur = st.number_input("Wage (Euros): ", min_value=0, max_value = 1000000000, value=0)
    movement_reactions = st.number_input("Movement Reaction: ", min_value=0, max_value = 99, value=0)
    mentality_composure = st.number_input("Mentality Composure: ", min_value=0, max_value = 99, value=0)

    if st.button("Predict"):
        # Create a DataFrame with the input features
        df = pd.DataFrame({
            'potential': [potential],
            'value_eur': [value_eur],
            'wage_eur': [wage_eur],
            'movement_reactions': [movement_reactions],
            'mentality_composure': [mentality_composure]
        })
        
        # Transform the input data
        scaled_df = scaler.transform(df)
        
        # Predict using the model with scaled data
        prediction = model.predict(scaled_df)
        output = prediction[0]
        

        confidence = 1 - abs(output - 75) / 75
        confidence_level = f"{confidence:.2f}%"

        st.success(f'Predicted Player Rating: {output:.0f} (Confidence: {confidence_level})')


if __name__ == "__main__":
    main()