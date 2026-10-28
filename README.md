from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB, BernoulliNB
from sklearn.metrics import classification_report, accuracy_score

documents = [
    ("NASA launched a new satellite to study the solar system", "space"),
    ("The rocket reached orbit after a successful launch", "space"),
    ("Astronauts on the space station conducted experiments", "space"),
    ("The telescopes captured images of a distant galaxy", "space"),
    ("Mission control confirmed the spacecraft landed safely", "space"),
    ("The shuttle carried a new crew to the orbital station", "space"),
    ("The goalie made a great save in the final period", "hockey"),
    ("The team won the championship after a hard fought game", "hockey"),
    ("The player scored a hat trick in last night's match", "hockey"),
    ("The coach praised the defense after the win", "hockey"),
    ("The hockey team advanced to another round", "hockey"),
    ("The arena was full for the playoff game", "hockey"),
    ("The president signed the bill into law today", "politics"),
    ("Voters debated the new law at the town hall meeting", "politics"),
    ("The committee reviewed the proposed legislation", "politics"),
    ("The election results were announced late last night", "politics"),
    ("The graphics card improves image processing significantly", "graphics"),
    ("The software renders 3D models using ray tracing", "graphics"),
    ("The animation studio upgraded its rendering pipeline", "graphics"),
    ("The GPU accelerated the video rendering task", "graphics"),
    ("The designer used the software to create 3D graphics", "graphics")
]

texts = [d[0] for d in documents]
labels = [d[1] for d in documents]

X_train_text, X_test_text, y_train, y_test = train_test_split(
    texts, labels, test_size=0.25, random_state=42, stratify=labels
)

vectorizer = CountVectorizer(stop_words='english')
X_train = vectorizer.fit_transform(X_train_text)
X_test = vectorizer.transform(X_test_text)

models = {
    "Multinomial NB": MultinomialNB(),
    "Bernoulli NB": BernoulliNB()
}

for name, model in models.items():
    model.fit(X_train, y_train)
    pred = model.predict(X_test)
    print(f"=== {name} ===")
    print(classification_report(y_test, pred, zero_division=0))
    print(f"Accuracy: {accuracy_score(y_test, pred):.3f}")
