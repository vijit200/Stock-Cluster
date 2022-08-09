from cProfile import label
import numpy as np
from flask import Flask,request,render_template
import pickle
model = pickle.load(open('model.pkl','rb'))
process = pickle.load(open('process.pkl','rb'))

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict',methods = ['POST'])

def predict():
    label = {0:'cluster 0',1:'cluster 1',2:'cluster 2',3:'cluster 3',4:'cluster 4'}
    if request.method == 'POST':
        SP500 = float(request.form['SP500'])
        NASDAQ_ADI = float(request.form['NASDAQ_ADI'])
        NASDAQ_AMD= float(request.form['NASDAQ_AMD'])
        NASDAQ_INCY = float(request.form['NASDAQ_INCY'])
        NYSE_EXR = float(request.form['NYSE_EXR'])
        l = [SP500,NASDAQ_ADI,NASDAQ_AMD,NASDAQ_INCY,NYSE_EXR]
        l1 = np.array(l)
        s = l1.reshape(1,-1)
        p = process.transform(s)
        model_eval = model.predict(p)[0]
        return  render_template('index.html',predection_eval = label[model_eval])

if __name__ == '__main__':
    app.run(debug=True)