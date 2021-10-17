import torch
from flask import Flask, request, jsonify, render_template
import os
from flask_cors import CORS, cross_origin
import pandas as pd

from com_in_ineuron_ai_utils.utils import decodeImage
# from predict import dogcat

os.putenv('LANG', 'en_US.UTF-8')
os.putenv('LC_ALL', 'en_US.UTF-8')

application = Flask(__name__)
CORS(application)


# @cross_origin()
class ClientApp:
    def __init__(self):
        self.filename = "inputImage.jpg"
        # self.classifier = dogcat(self.filename)


@application.route("/", methods=['GET'])
@cross_origin()
def home():
    return render_template('index.html')


@application.route("/predict", methods=['POST'])
@cross_origin()
def predictRoute():
    # image = request.json['image']
    # decodeImage(image, clApp.filename)
    # result = clApp.classifier.predictiondogcat()
    # return jsonify(result)

    #
    # model = torch.hub.load('ultralytics/yolov5', 'custom', path='C:/Users/Sneha/Downloads/Cat_Dog-main/best.pt',
    #                        force_reload=True)
    # img ='C:/Users/Sneha/Downloads/Cat_Dog-main/test/BloodImage_00038_jpg.rf.ffa23e4b5b55b523367f332af726eae8.jpg'
    # result= model(img)
    import subprocess
    test = subprocess.run("python detect.py --source 0", stdout=subprocess.PIPE)
    #Popen(["ping", "-W", "2", "-c", "1", "192.168.1.70"], stdout=subprocess.PIPE)
    prediction = test.stdout #.communicate()[0]
    prediction = str(prediction).split("FPS).")[1]
    # print(type(str(prediction).split("\"")[0]))
    print(prediction)
    #     #
    #     #
    #     #
    #     df= pd.DataFrame(result.pandas().xyxy[0])
    #     bloodtype=df.groupby('name').count()
    #     pr=bloodtype.iloc[:,-1:]
    #     pri=pr.to_dict()
    #
    #     # bloodtye = request.json[pri]
    #     # # # decodeImage(image, clApp.filename)
    #     result = clApp.classifier.predictiondogcat()
    #     # # return jsonify(result)


    # print(result)



    # print(model)

    # return jsonify(result)
    p = prediction#.replace("\\r",'').replace("\\n",'')
    # p = p.split(", Done.")
    print("Mera wala print : ",p)
    return (jsonify([{ "image" : p}]))


if __name__ == "__main__":
    clApp = ClientApp()
    application.run(debug=True)
    # app.run(debug=True)
