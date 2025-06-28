from flask import Flask, render_template, request, redirect, url_for, session, flash, Response
from flask_sqlalchemy import SQLAlchemy
from werkzeug.security import generate_password_hash, check_password_hash
import joblib
import numpy as np
import os

app = Flask(__name__)
app.secret_key = 'super-secret-key'

# Configuration DB
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///estimations.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
db = SQLAlchemy(app)

# ------------------ MODELS -------------------
class User(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    email = db.Column(db.String(120), unique=True, nullable=False)
    password_hash = db.Column(db.String(128), nullable=False)

    def set_password(self, password):
        self.password_hash = generate_password_hash(password)

    def check_password(self, password):
        return check_password_hash(self.password_hash, password)

class Estimation(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    longitude = db.Column(db.Float)
    latitude = db.Column(db.Float)
    housing_median_age = db.Column(db.Float)
    total_rooms = db.Column(db.Float)
    total_bedrooms = db.Column(db.Float)
    population = db.Column(db.Float)
    households = db.Column(db.Float)
    median_income = db.Column(db.Float)
    ocean_proximity = db.Column(db.String(20))
    prediction = db.Column(db.Float)
    created_at = db.Column(db.DateTime, server_default=db.func.now())

with app.app_context():
    db.create_all()

# ------------------ MODEL ML -------------------
try:
    model = joblib.load('model.pkl')
except:
    model = None

# ------------------ ROUTES AUTH -------------------
@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        email = request.form['email']
        password = request.form['password']
        if User.query.filter_by(email=email).first():
            flash('❌ Email déjà utilisé.')
            return redirect(url_for('register'))

        new_user = User(email=email)
        new_user.set_password(password)
        db.session.add(new_user)
        db.session.commit()
        flash('✅ Compte créé ! Connectez-vous.')
        return redirect(url_for('login'))

    return render_template('register.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        email = request.form['email']
        password = request.form['password']
        user = User.query.filter_by(email=email).first()
        if user and user.check_password(password):
            session['user'] = email
            flash(f'✅ Bienvenue {email} !')
            return redirect(url_for('home_page'))
        flash('❌ Identifiants incorrects.')
    return render_template('login.html')

@app.route('/logout')
def logout():
    session.pop('user', None)
    flash('✅ Vous êtes déconnecté.')
    return redirect(url_for('login'))

# ------------------ ROUTES APP -------------------
@app.route('/')
def index():
    return redirect(url_for('login'))

@app.route('/home')
def home_page():
    if 'user' not in session:
        return redirect(url_for('login'))
    return render_template('home.html')

@app.route('/predict_page')
def predict_page():
    if 'user' not in session:
        return redirect(url_for('login'))
    return render_template('prédicteur.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'user' not in session:
        return redirect(url_for('login'))
    if model is None:
        return "Erreur : Le modèle est introuvable."

    try:
        longitude = float(request.form['longitude'])
        latitude = float(request.form['latitude'])
        housing_median_age = float(request.form['housing_median_age'])
        total_rooms = float(request.form['total_rooms'])
        total_bedrooms = float(request.form['total_bedrooms'])
        population = float(request.form['population'])
        households = float(request.form['households'])
        median_income = float(request.form['median_income'])
        ocean = request.form['ocean_proximity']

        ocean_map = {
            'INLAND': [1, 0, 0, 0],
            'ISLAND': [0, 1, 0, 0],
            'NEAR BAY': [0, 0, 1, 0],
            'NEAR OCEAN': [0, 0, 0, 1]
        }
        ocean_features = ocean_map.get(ocean, [0, 0, 0, 0])

        features = np.array([
            longitude, latitude, housing_median_age, total_rooms,
            total_bedrooms, population, households, median_income
        ] + ocean_features).reshape(1, -1)

        prediction = model.predict(features)[0]
        prediction_rounded = round(float(prediction), 2)

        estimation = Estimation(
            longitude=longitude,
            latitude=latitude,
            housing_median_age=housing_median_age,
            total_rooms=total_rooms,
            total_bedrooms=total_bedrooms,
            population=population,
            households=households,
            median_income=median_income,
            ocean_proximity=ocean,
            prediction=prediction_rounded
        )
        db.session.add(estimation)
        db.session.commit()

        return render_template('prédicteur.html', prediction=prediction_rounded)

    except Exception as e:
        return f"Erreur lors de la prédiction : {str(e)}"

@app.route('/historique')
def historique():
    if 'user' not in session:
        return redirect(url_for('login'))
    estimations = Estimation.query.order_by(Estimation.created_at.desc()).all()
    return render_template('historique.html', estimations=estimations)

@app.route('/map')
def map_page():
    if 'user' not in session:
        return redirect(url_for('login'))
    estimations = Estimation.query.order_by(Estimation.created_at.desc()).all()
    return render_template('map.html', estimations=estimations)



@app.route('/export_csv')
def export_csv():
    if 'user' not in session:
        return redirect(url_for('login'))
    estimations = Estimation.query.order_by(Estimation.created_at.desc()).all()
    def generate():
        yield 'ID,Longitude,Latitude,HousingMedianAge,TotalRooms,TotalBedrooms,Population,Households,MedianIncome,OceanProximity,Prediction,CreatedAt\n'
        for e in estimations:
            yield f'{e.id},{e.longitude},{e.latitude},{e.housing_median_age},{e.total_rooms},{e.total_bedrooms},{e.population},{e.households},{e.median_income},{e.ocean_proximity},{e.prediction},{e.created_at}\n'
    return Response(generate(), mimetype='text/csv', headers={'Content-Disposition': 'attachment; filename=historique_predictions.csv'})

@app.route('/delete_estimations', methods=['POST'])
def delete_estimations():
    Estimation.query.delete()
    db.session.commit()
    flash("Toutes les estimations ont été supprimées.")
    return redirect(url_for('historique'))

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    app.run(debug=True, host="0.0.0.0", port=port)
