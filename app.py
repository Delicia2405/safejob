from flask import Flask, request, render_template, session
import sqlite3
import subprocess

from predict import predict
from explain import explain
from salary_engine import salary_check
from heatmap import get_city_risk

app = Flask(__name__)
app.secret_key = "safejob_secret_2024"


def combine_scores(nlp_score, salary_result, city_risk_result):
    if salary_result["anomaly"]:
        salary_score = min(abs(salary_result["z_score"]) / 4, 1.0)
    else:
        salary_score = 0.0

    location_score = city_risk_result.get("risk_index", 50) / 100

    final_score = (0.40 * nlp_score) + (0.20 * salary_score) + (0.40 * location_score)
    final_score = round(final_score, 4)

    if final_score >= 0.75:
        verdict = "CRITICAL"
    elif final_score >= 0.55:
        verdict = "HIGH"
    elif final_score >= 0.35:
        verdict = "MEDIUM"
    else:
        verdict = "LOW"

    return final_score, verdict


def init_db():
    conn = sqlite3.connect('reports.db')
    c = conn.cursor()
    c.execute('''
        CREATE TABLE IF NOT EXISTS reports (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            job_description TEXT,
            recruiter_contact TEXT,
            platform TEXT,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    conn.commit()
    conn.close()


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/check-job')
def check_job_page():
    return render_template('check.html')


@app.route('/about')
def about():
    return render_template('about.html')


@app.route('/report-portal')
def report_portal():
    return render_template('report_portal.html')


@app.route('/check', methods=['POST'])
def check_job():
    job_title = request.form['job_title']
    salary = float(request.form['salary'])
    city = request.form['city']
    description = request.form['description']

    nlp_score = predict(description)
    explanation = explain(description)
    salary_result = salary_check(job_title, salary)
    salary_result['offered_salary'] = salary
    city_result = get_city_risk(city)
    final_score, verdict = combine_scores(nlp_score, salary_result, city_result)

    session['results'] = {
        'final_score': final_score,
        'verdict': verdict,
        'explanation': explanation,
        'salary_result': salary_result,
        'city_result': city_result,
        'description': description,
        'job_title': job_title,
    }

    return render_template('result_score.html',
        final_score=final_score,
        verdict=verdict,
        job_title=job_title
    )


@app.route('/results/words')
def results_words():
    r = session.get('results', {})
    return render_template('result_words.html',
        explanation=r.get('explanation', []),
        verdict=r.get('verdict', 'MEDIUM')
    )


@app.route('/results/salary')
def results_salary():
    r = session.get('results', {})
    return render_template('result_salary.html',
        salary_result=r.get('salary_result', {}),
        verdict=r.get('verdict', 'MEDIUM')
    )


@app.route('/results/location')
def results_location():
    r = session.get('results', {})
    return render_template('result_location.html',
        city_result=r.get('city_result', {}),
        verdict=r.get('verdict', 'MEDIUM'),
        final_score=r.get('final_score', 0),
        description=r.get('description', ''),
        job_title=r.get('job_title', '')
    )


@app.route('/report', methods=['POST'])
def report_job():
    job_description = request.form['job_description']
    recruiter_contact = request.form.get('recruiter_contact', '')
    platform = request.form.get('platform', '')

    conn = sqlite3.connect('reports.db')
    c = conn.cursor()
    c.execute('''
        INSERT INTO reports (job_description, recruiter_contact, platform)
        VALUES (?, ?, ?)
    ''', (job_description, recruiter_contact, platform))
    conn.commit()

    count = c.execute('SELECT COUNT(*) FROM reports').fetchone()[0]
    conn.close()

    if count % 30 == 0:
        subprocess.Popen(['python', 'train_model.py'])
        print(f"🔄 Retraining triggered at {count} reports!")

    return render_template('report_success.html')


init_db()

if __name__ == '__main__':
    app.run(debug=True)