from flask import Flask,render_template, request
import kmeans as km

app = Flask(__name__)

@app.route("/", methods = ['GET','POST'])
def home():
    clstr = ''
    if request.method =='POST':
        hrs = request.form['submit'] 
        cluster_pred = km.flood_cluster(hrs)
        clstr = cluster_pred 
    
    return render_template('home.html', fld = clstr)

if __name__=="__main__":
    app.run(debug=True)