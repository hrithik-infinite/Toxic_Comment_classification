from flask import Flask, render_template, request  
import pickle

app = Flask(__name__)
vekt = pickle.load(open('vekt.pkl', 'rb'))

m1 = pickle.load(open('m1.pkl', 'rb'))
m2 = pickle.load(open('m2.pkl', 'rb'))
m3 = pickle.load(open('m3.pkl', 'rb'))
m4 = pickle.load(open('m4.pkl', 'rb'))
m5 = pickle.load(open('m5.pkl', 'rb'))
m6 = pickle.load(open('m6.pkl', 'rb'))


@app.route("/")
def home():
    return render_template('s2.html')

@app.route("/predict", methods = ['POST'])

def predict():
    ip = request.form['text']
    ip_data = [ip]

    vect = vekt.transform(ip_data)
    p_tox = m1.predict_proba(vect)[:,1]
    p_sev = m2.predict_proba(vect)[:,1]
    p_obs = m3.predict_proba(vect)[:,1]
    p_thr = m4.predict_proba(vect)[:,1]
    p_ins = m5.predict_proba(vect)[:,1]
    p_ide = m6.predict_proba(vect)[:,1]
    
    p_ptox = round(p_tox[0], 3)
    p_psev = round(p_sev[0], 3)
    p_pobs = round(p_obs[0], 3)
    p_pins = round(p_ins[0], 3)
    p_pthr = round(p_thr[0], 3)
    p_pide = round(p_ide[0], 3)

    return render_template('s2.html',
    tox = 'Toxic Percentage - {:.2f}%'.format(p_ptox * 100),
    sev = 'Severe Toxic Percentage - {:.2f}%'.format(p_psev * 100),
    obs = 'Obscene Percentage - {:.2f}%'.format(p_pobs * 100),
    ins = 'Insult Percentage - {:.2f}%'.format(p_pins * 100),
    thr = 'Threat Percentage - {:.2f}%'.format(p_pthr * 100),
    ide = 'Identity Hate Percentage - {:.2f}%'.format(p_pide * 100))
if __name__ == '__main__':
    app.run(debug = True)

