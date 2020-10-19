import pandas as pd

from flask import Flask, jsonify, request

from models.prediction import make_prediction


app = Flask(__name__)


@app.route('/predict', methods=['POST'])
def apicall():
    try:
        to_predict = request.get_json(force=True)
        if isinstance(to_predict, dict): to_predict = [to_predict];
        to_predict = pd.DataFrame(to_predict)
        to_predict.set_index('ID', inplace=True)
    except Exception as exception:
        raise exception

    if to_predict.empty:
        return bad_request()
    else:
        predictions = pd.Series(make_prediction(models_dir='../models/', to_predict=to_predict)[:, 1],
                                index=to_predict.index).to_dict()
        responses = jsonify(predictions=predictions)
        responses.status_code = 200
        return responses


@app.errorhandler(400)
def bad_request(error=None):
    message = {
        'status': 400,
        'message': 'Bad Request: ' + request.url + ' --> Please check your data payload...'
    }
    resp = jsonify(message)
    resp.status_code = 400

    return resp


if __name__ == '__main__':
    app.run()