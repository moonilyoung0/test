from flask import Flask, render_template, request

import test

app = Flask(__name__)

@app.route("/", methods=['GET','POST'])
def test_ml():
    if request.method == "POST":
        video_url = request.form["video_url"]
        res, res2 = test.test(video_url)
    return render_template("index.html")

if __name__ == "__main__":
    app.run(debug=True)