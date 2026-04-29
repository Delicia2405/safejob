import numpy as np

salary_data = {
    "software engineer": (40000, 15000),
    "data analyst": (35000, 12000),
    "teacher": (25000, 8000),
    "construction worker": (15000, 5000),
    "house maid": (12000, 4000),
    "driver": (18000, 6000),
    "sales executive": (22000, 7000),
    "nurse": (30000, 10000),
    "security guard": (14000, 4000),
    "overseas helper": (12000, 4000),
    "helper": (12000, 4000),
    "domestic worker": (12000, 4000),
}


def get_job_category(title):
    title = title.lower()
    for key in salary_data:
        if key in title:
            return key
    return "software engineer"


def salary_check(job_title: str, offered_salary: float):
    category = get_job_category(job_title)

    mean, std = salary_data[category]

    z_score = (offered_salary - mean) / std

    anomaly = abs(z_score) > 2

    return {
        "category": category,
        "market_avg": mean,
        "z_score": round(z_score, 2),
        "anomaly": anomaly
    }


# test
if __name__ == "__main__":
    print(salary_check("Software Engineer", 100000))