from flask import Flask, request, render_template, redirect, url_for, session, flash, Response
from flask_sqlalchemy import SQLAlchemy
import numpy as np
import joblib
import csv
import os

app = Flask(__name__)
app.secret_key = 'mot-de-passe-super-secret'

# 🔗 Connexion à MySQL (modifie le mot de passe si besoin)
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///estimations.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
db = SQLAlchemy(app)

# 📦 Charger le modèle entraîné
model = joblib.load('model.pkl')

# 📋 Modèle SQLAlchemy = structure de la table estimation
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

# 🏠 Page d'accueil
@app.route('/', methods=['GET'])
def home():
    return render_template('design.html')

# 🎯 Route de prédiction
@app.route('/predict', methods=['POST'])
def predict():
    try:
        # 1. Récupérer les données du formulaire
        longitude = float(request.form['longitude'])
        latitude = float(request.form['latitude'])
        housing_median_age = float(request.form['housing_median_age'])
        total_rooms = float(request.form['total_rooms'])
        total_bedrooms = float(request.form['total_bedrooms'])
        population = float(request.form['population'])
        households = float(request.form['households'])
        median_income = float(request.form['median_income'])
        ocean = request.form['ocean_proximity']

        # 2. Codage one-hot
        ocean_map = {
            'INLAND': [1, 0, 0, 0],
            'ISLAND': [0, 1, 0, 0],
            'NEAR BAY': [0, 0, 1, 0],
            'NEAR OCEAN': [0, 0, 0, 1]
        }
        ocean_features = ocean_map.get(ocean, [0, 0, 0, 0])

        # 3. Prédiction
        features = np.array([
            longitude, latitude, housing_median_age, total_rooms,
            total_bedrooms, population, households, median_income
        ] + ocean_features).reshape(1, -1)

        prediction = model.predict(features)[0]
        prediction_rounded = round(float(prediction), 2)

        # 4. Enregistrement dans la base MySQL
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

        # 5. Affichage
        return render_template('design.html', prediction=prediction_rounded)

    except Exception as e:
        return f"Erreur : {str(e)}"

# 📄 Historique des prédictions
@app.route('/historique')
def historique():
    if 'user' not in session:
        return redirect(url_for('login'))

    estimations = Estimation.query.order_by(Estimation.created_at.desc()).all()
    return render_template('historique.html', estimations=estimations)

# 🔐 Connexion utilisateur
@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        email = request.form['email']
        password = request.form['password']

        # Pour le test : login fixe
        if email == 'admin@gmail.com' and password == '123456':
            session['user'] = email
            return redirect(url_for('historique'))
        else:
            flash('❌ Identifiants incorrects')

    return render_template('login.html')

# 🔓 Déconnexion
@app.route('/logout')
def logout():
    session.pop('user', None)
    flash("Vous êtes déconnecté.")
    return redirect(url_for('login'))

# ⬇️ Export CSV
@app.route('/export_csv')
def export_csv():
    if 'user' not in session:
        return redirect(url_for('login'))

    estimations = Estimation.query.order_by(Estimation.created_at.desc()).all()

    def generate():
        yield 'ID,Longitude,Latitude,HousingMedianAge,TotalRooms,TotalBedrooms,Population,Households,MedianIncome,OceanProximity,Prediction,CreatedAt\n'
        for e in estimations:
            yield f'{e.id},{e.longitude},{e.latitude},{e.housing_median_age},{e.total_rooms},{e.total_bedrooms},{e.population},{e.households},{e.median_income},{e.ocean_proximity},{e.prediction},{e.created_at}\n'

    return Response(
        generate(),
        mimetype='text/csv',
        headers={'Content-Disposition': 'attachment; filename=historique_predictions.csv'}
    )
# Supprimer toutes les estimations
@app.route('/delete_estimations', methods=['POST'])
def delete_estimations():
    Estimation.query.delete()
    db.session.commit()
    flash("Toutes les estimations ont été supprimées.")
    return redirect(url_for('historique'))

# 🚀 Lancement
if __name__ == '__main__':
    with app.app_context():
        db.create_all()
    app.run(debug=True, host="0.0.0.0", port=int(os.environ.get("PORT", 5000)))


