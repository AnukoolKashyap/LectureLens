"""Build a three-video sample course for offline testing.

    python sample_data/make_course_sample.py

The three transcripts deliberately overlap: the learning rate is introduced in
video 1, revisited in video 2 and tuned in video 3. That overlap is what makes
cross-video retrieval worth testing — a course should be able to answer "where
is the learning rate covered?" with more than one place.
"""
from __future__ import annotations

import json
from pathlib import Path

LESSONS = [
    (
        "course_01_gradient_descent",
        "Lesson 1 - Gradient Descent and the Loss Surface",
        """
        welcome to the first lesson of this short course on training neural networks.
        today we only care about one question, which is how a network gets to the right
        numbers for its weights. the answer is gradient descent. start with the loss
        function. the loss is a single number telling you how wrong the network is right
        now. for regression we use mean squared error, which averages the squared
        differences between prediction and truth. for classification we use cross
        entropy, which punishes a confident wrong answer far more harshly than a
        hesitant one. now picture every weight as a coordinate and the loss as the
        height of a landscape above those coordinates. training is walking downhill on
        that landscape until you reach a valley. the direction downhill comes from the
        gradient, which is the vector of partial derivatives of the loss with respect to
        each weight. the gradient points uphill, so we step in the negative gradient
        direction. the size of that step is the learning rate. we will come back to the
        learning rate properly in lesson three, because choosing it badly is the single
        most common reason training fails. for now just remember that it controls how
        far you move on each step. next lesson we will look at how the gradient is
        actually computed for a network with millions of weights.
        """,
    ),
    (
        "course_02_backpropagation",
        "Lesson 2 - Backpropagation and the Chain Rule",
        """
        in the previous lesson we said that the gradient tells us which way is downhill.
        we did not say how to compute it. that is what backpropagation does, and it is
        the subject of this lesson. backpropagation is the chain rule applied
        systematically, from the output layer backwards to the input layer. you compute
        the error at the output, then you ask how much each weight in the layer before
        contributed to that error, and you keep pushing that question backwards through
        the network. the reason it is efficient is reuse. the intermediate results you
        need for one layer are exactly the results you already computed for the layer
        after it, so nothing is calculated twice. without that reuse you would need a
        separate pass for every single weight, which for a modern network is completely
        impractical. a common confusion is between backpropagation and gradient descent.
        they are not the same thing. backpropagation computes the gradient. gradient
        descent is what uses the gradient to update the weights, and the learning rate
        is what scales that update. one more practical point before we finish. in
        practice nobody runs this over the whole dataset before taking a step. instead
        we use stochastic gradient descent, estimating the gradient from a small batch
        of maybe thirty two or sixty four examples and stepping immediately. the
        estimate is noisy but you take thousands of steps instead of one, and the noise
        actually helps you escape shallow local minima.
        """,
    ),
    (
        "course_03_learning_rate",
        "Lesson 3 - Choosing and Scheduling the Learning Rate",
        """
        this lesson is entirely about the learning rate, which we have mentioned twice
        and kept postponing. the learning rate decides how far you move along the
        negative gradient on each step. if it is too small, training crawls and you run
        out of patience or compute budget before the loss comes down. if it is too
        large, you overshoot the valley, bounce from wall to wall, and the loss diverges
        to infinity. you will see that in your logs as nan values appearing. so how do
        you pick one. the practical answer is a learning rate range test. you start with
        a very small rate, increase it steadily over a few hundred iterations, and plot
        the loss against the rate. the best choice is usually just below the point where
        the loss starts climbing. after that, use a schedule. it is common to start
        relatively large and decay over training, because early on you want to cover
        ground quickly and later you want to settle into the bottom of the valley rather
        than bouncing around it. cosine decay and step decay are both fine, and the
        specific schedule matters far less than having one at all. finally, the
        optimisers. momentum keeps a running average of past gradients so you build
        speed in consistent directions and damp oscillation across a narrow valley.
        rmsprop scales the step per parameter so rarely updated parameters still move.
        adam combines both, which is why it is the default in most codebases. it is not
        always the best, but it is almost never terrible, and that is a good property
        for a default. that is the end of the course.
        """,
    ),
]


def build(text: str, words_per_segment: int = 20, seconds: float = 8.0) -> list:
    words = text.split()
    segments, bucket, start = [], [], 0.0
    for word in words:
        bucket.append(word)
        if len(bucket) >= words_per_segment:
            segments.append({"text": " ".join(bucket), "start": round(start, 1),
                             "duration": seconds})
            start += seconds
            bucket = []
    if bucket:
        segments.append({"text": " ".join(bucket), "start": round(start, 1),
                         "duration": seconds})
    return segments


if __name__ == "__main__":
    here = Path(__file__).resolve().parent
    written = []
    for video_id, title, text in LESSONS:
        payload = {
            "video_id": video_id,
            "url": "local://%s.json" % video_id,
            "title": title,
            "author": "Sample course (bundled with the project)",
            "segments": build(" ".join(text.split())),
        }
        out = here / ("%s.json" % video_id)
        out.write_text(json.dumps(payload, indent=2, ensure_ascii=False),
                       encoding="utf-8")
        written.append((out.name, len(payload["segments"])))

    for name, n in written:
        print("Wrote %s — %d segments" % (name, n))
    print("\nRun the course offline with:")
    print("  python cli.py --local-course sample_data/course_01_gradient_descent.json "
          "sample_data/course_02_backpropagation.json "
          "sample_data/course_03_learning_rate.json --offline")
