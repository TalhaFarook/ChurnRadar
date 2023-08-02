import pickle, pandas
from flask import Flask, request, jsonify

app = Flask(__name__)

# Loading the prepared model's pickle file
with open('chum_predict.pkl', 'rb') as file:
    model, scaler, encoder = pickle.load(file)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.json

        #Converting the JSON format to pandas dataframe for our model to read 
        input_data = pandas.DataFrame([data])

        #Applying the appropriate transformation
        input_data[['Age', 'Service Length', 'Monthly Charges', 'Total Charges']] = scaler.transform(input_data[['Age', 'Service Length', 'Monthly Charges', 'Total Charges']])

        input_data[['Gender', 'Contract Type']] = encoder.transform(input_data[['Gender', 'Contract Type']])

        #Rearranging the dataframe to match the column order when our model was fit
        input_data = input_data[['Gender', 'Age', 'Service Length', 'Contract Type', 'Monthly Charges',
       'Total Charges']]

        prediction = model.predict(input_data)

        if prediction[0] == 'Yes':
            return jsonify({'Churn': 'Yes'})
        elif prediction[0] == 'No':
            return jsonify({'Churn': 'No'})

    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run(debug=True)