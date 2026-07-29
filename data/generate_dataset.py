"""
generate_dataset.py
--------------------
Generates a synthetic SMS Spam/Ham dataset for the project.

NOTE: This environment has no internet access, so we cannot download the
real "SMS Spam Collection" dataset from Kaggle/UCI directly here.
This script creates a realistic, template-based synthetic dataset instead,
so the ENTIRE pipeline (cleaning -> training -> evaluation -> app) runs
end-to-end right now.

>>> FOR YOUR FINAL SUBMISSION <<<
Replace data/spam.csv with the real dataset for better real-world accuracy:
    1. Go to: https://www.kaggle.com/datasets/uciml/sms-spam-collection-dataset
    2. Download "spam.csv"
    3. Place it in the data/ folder (keep columns named 'label' and 'message')
Everything else (train.py, app.py) will work unchanged.
"""

import random
import csv

random.seed(42)

# ---------------------------------------------------------------------
# Building blocks for HAM (normal) messages
# ---------------------------------------------------------------------
ham_templates = [
    "Hey, are we still meeting at {time} today?",
    "Can you pick up milk on your way home?",
    "Happy birthday! Hope you have a great day.",
    "I'll be there in {num} minutes, traffic is bad.",
    "Don't forget the meeting tomorrow at {time}.",
    "Thanks for helping me yesterday, really appreciate it.",
    "What time does the movie start tonight?",
    "Mom said dinner will be ready by {time}.",
    "Can we reschedule our call to {time}?",
    "I finished the assignment, sending it to you now.",
    "Let's catch up this weekend, it's been a while.",
    "Are you coming to the party on Saturday?",
    "Please send me the notes from today's class.",
    "I reached home safely, thanks for dropping me.",
    "How was your exam? Hope it went well.",
    "Let me know once you land at the airport.",
    "The project deadline got extended to next {day}.",
    "Can you send me your address again?",
    "I'm running late, start without me.",
    "Great job on the presentation today!",
    "Do you want to grab coffee tomorrow?",
    "I left my charger at your place, can I pick it up?",
    "The weather looks nice today, want to go for a walk?",
    "Congrats on the new job! So happy for you.",
    "Can you help me move some boxes this weekend?",
    "I watched that movie you recommended, it was great.",
    "Please call me when you're free, nothing urgent.",
    "Just checking in, how have you been?",
    "See you at the gym at {time}?",
    "I'll send the report by end of day.",
]

names = ["Rahul", "Priya", "Aman", "Sara", "John", "Meera", "Ravi", "Anjali"]
times = ["9am", "10:30am", "6pm", "7:15pm", "noon", "5pm", "8am"]
days = ["Monday", "Friday", "Wednesday"]

# ---------------------------------------------------------------------
# Building blocks for SPAM messages
# ---------------------------------------------------------------------
spam_templates = [
    "Congratulations! You have WON a {prize} worth Rs.{amount}. Click here to claim now: {link}",
    "URGENT! Your account has been suspended. Verify now at {link} to avoid deactivation.",
    "You have been selected for a FREE {prize}!! Call {phone} now to claim your prize.",
    "WINNER!! As a valued customer you have won a {prize}. Text WIN to {phone} to collect.",
    "Get a loan of Rs.{amount} instantly with ZERO paperwork. Apply now: {link}",
    "Your mobile number has won {amount} in the lucky draw! Claim before it expires: {link}",
    "FREE entry into our {amount} weekly prize draw just by texting WIN to {phone}.",
    "Limited time offer! Get {percent}% off on all products. Shop now: {link}",
    "Dear customer, your bank account will be blocked. Update KYC immediately: {link}",
    "You've been chosen to receive a free {prize}. Reply YES to claim your gift now.",
    "Cash prize of Rs.{amount} is waiting for you! Click {link} to claim instantly.",
    "Congrats! You are pre-approved for a credit card with Rs.{amount} limit. Apply: {link}",
    "Hot singles in your area want to chat with you now! Click {link}",
    "Your parcel could not be delivered. Pay Rs.{amount} customs fee here: {link}",
    "Earn Rs.{amount} per day working from home! No investment needed. Join now: {link}",
    "FINAL NOTICE: Your subscription payment failed. Update details at {link} immediately.",
    "You have an unclaimed refund of Rs.{amount}. Claim it now before it expires: {link}",
    "Click here for a FREE {prize} + free shipping, limited stock available: {link}",
    "Your number has been randomly selected to win an iPhone! Claim at {link}",
    "Act now! Investment opportunity guaranteed to double your money in {num} days.",
]

prizes = ["iPhone 15", "gift voucher", "laptop", "holiday package", "smartwatch", "cash reward"]
links = ["bit.ly/claim-now", "tinyurl.com/win-prize", "secure-verify.info", "offer-link.co", "get-reward.net"]
phones = ["09876543210", "07777888999", "08001234567"]
amounts = ["5000", "10000", "50000", "1,00,000", "2500"]
percents = ["50", "70", "80", "90"]

def fill(template):
    return template.format(
        time=random.choice(times),
        num=random.randint(2, 45),
        day=random.choice(days),
        prize=random.choice(prizes),
        amount=random.choice(amounts),
        link=random.choice(links),
        phone=random.choice(phones),
        percent=random.choice(percents),
    )

def make_dataset(n_ham=1300, n_spam=250):
    rows = []
    for _ in range(n_ham):
        msg = fill(random.choice(ham_templates))
        # occasionally personalize with a name
        if random.random() < 0.3:
            msg = f"{random.choice(names)}, " + msg[0].lower() + msg[1:]
        rows.append(("ham", msg))

    for _ in range(n_spam):
        msg = fill(random.choice(spam_templates))
        rows.append(("spam", msg))

    random.shuffle(rows)
    return rows

if __name__ == "__main__":
    rows = make_dataset()
    out_path = "data/spam.csv" if __name__ == "__main__" else "spam.csv"
    with open("data/spam.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["label", "message"])
        writer.writerows(rows)
    print(f"Generated {len(rows)} messages -> data/spam.csv")
    spam_count = sum(1 for r in rows if r[0] == "spam")
    print(f"Spam: {spam_count}, Ham: {len(rows) - spam_count}")
