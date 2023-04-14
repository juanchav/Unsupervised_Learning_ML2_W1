from flask_wtf import FlaskForm
from wtforms import StringField,SubmitField

class PredecirForm(FlaskForm):
    Record=StringField('Record')
    submit = SubmitField('Classify')

 