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

if __name__=="__main__":
    app.run(debug=True,port=8000,host="0.0.0.0")
