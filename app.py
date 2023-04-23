from flask import Flask, jsonify, request
import numpy as np
import pickle


app = Flask(__name__)

##load a file
with open('rf_model.pkl','rb') as f:
    rf_model = pickle.load(f)


@app.route('/predict_rejection', methods=['GET','POST'])
def predict():
    # Get the request data
    data = request.form
    if request.method =='POST':
        Amount = float(data['Amount'])
        B2B = int(data['B2B'])
        Category = int(data['Category'])
        Fulfilment = int(data['Fulfilment'])
        Size = int(data['Size'])
        region = int(data['region'])

    # Create a test array with the input features
    test_array = np.array([[Amount, B2B, Category, Fulfilment, Size, region]])

    # Predict the rejection using the `predicted_rejection` function
    def predicted_rejection(test_array):
        rejection = rf_model.predict_rejection(test_array)
        rejection = np.around(rejection, 2)

        return rejection
    
    rejection = predicted_rejection(test_array)


    # Return the result as a JSON object
    return jsonify({'rejection': rejection[1]})
print('Product was rejected by customer due to high pricing')


if __name__ == '__main__':
    # Start the Flask application
    app.run(debug=True)