from flask import Flask,request, url_for, redirect, render_template
import pickle
import numpy as np
import sklearn.linear_model
from sklearn.feature_extraction.text import TfidfVectorizer

app = Flask(__name__,template_folder='templates')


model = pickle.load(open(r'D:\Opera Download\Spam mail\model.pkl', 'rb'))

@app.route('/')
def hello_world():
    return render_template("new.html")


@app.route('/predict.html',methods=['POST','GET'])
def predict():
    # text=[str(request.form.values())]
    # text=[str(request.form.AllKeys)]
    text = [value for key, value in request.form.items()]
    
    # m = ["Free!! Free!! Free!!"]
    # xvector = TfidfVectorizer(min_df =1,stop_words="english",lowercase =True)
    # print(text)
    xvector = pickle.load(open(r'D:\Opera Download\Spam mail\xvector.pickle', 'rb')) 
    d = xvector.transform(text)
    output = model.predict(d)
    # if(output[0]==1):print("SPAM")
    # else:print("HAM")


    if output[0]==1:
            return render_template('/predict.html',pred='This Mail {} is SPAM {}'.format(text,output))
    else:
            return render_template('/predict.html',pred='This Mail  {} is HAM Mail {}'.format(text,output))




    # int_features=[int(x) for x in request.form.values()]
    # final=[np.array(int_features)]
    # print(int_features)
    # print(final)
    # prediction=model.predict_proba(final)
    # output='{0:.{1}f}'.format(prediction[0][1], 2)

    # if output>str(0.5):
    #     return render_template('forest_fire.html',pred='Your Forest is in Danger.\nProbability of fire occuring is {}'.format(output),bhai="kuch karna hain iska ab?")
    # else:
    #     return render_template('forest_fire.html',pred='Your Forest is safe.\n Probability of fire occuring is {}'.format(output),bhai="Your Forest is Safe for now")


if __name__ == '__main__':
    app.run(debug=True)
