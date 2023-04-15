from flask import Flask,render_template,request,flash
from forms import PredecirForm
import os
from flask_bootstrap import Bootstrap
import joblib




secretkey=os.urandom(32)
app=Flask(__name__,instance_relative_config=False)
app.secret_key=secretkey
Bootstrap(app)


# Decorators
@app.route('/',methods=('GET','POST'))
def home():
    datos={}
    form=PredecirForm(request.form)

    if request.method=='POST':

        if form.validate_on_submit():
            record=form.Record.data

            flash(datos)
        else:
            flash('Please review the data entered','danger')

    return render_template("home.html",form=form)

    # Get Mnist
    (X_train, y_train), (X_test, y_test) = mnist.load_data()
    masktrain = (y_train == 0) | (y_train == 8)
    masktest = (y_test == 0) | (y_test == 8)
    X_train = X_train[masktrain]/255.0
    y_train = y_train[masktrain]
    X_test = X_test[masktest]/255.0
    y_test = y_test[masktest]
    X_train = X_train.reshape(X_train.shape[0] , X_train.shape[1]* X_train.shape[2])   
    X_test = X_test.reshape(X_test.shape[0] , X_test.shape[1]* X_test.shape[2])  
    # create a PCA object with 2 components
    pca = PCA(n_components=2)
    # fit the data
    pca.fit(X_train)
    # transform the data using the PCA object
    reduced_x_train = pca.transform(X_train)

if __name__=="__main__":
    app.run(debug=True,port=8000,host="0.0.0.0")
