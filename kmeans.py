import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2

def definition():
    df = pd.read_csv("data-kejadian-bencana-banjir-provinsi-dki-jakarta-2020-bulan-januari-maret_tes.csv")
    df1 = df.drop(['jumlah_meninggal','jumlah_hilang','jumlah_luka_berat','jumlah_tempat_pengungsian','nilai_kerugian','lama_genangan','jumlah_pengungsi_tertinggi'], axis=1)

    X = df1.iloc[:,4:11]
    y = df1.iloc[:,2]


    bestfeature = SelectKBest(score_func=chi2, k=4)
    fit = bestfeature.fit(X,y)
    dfscores = pd.DataFrame(fit.scores_)
    dfcolums = pd.DataFrame(X.columns)


    featureScores = pd.concat([dfcolums, dfscores],axis=1)
    featureScores.columns = ['Field', 'Score']
    print(featureScores.nlargest(10,'Score'))


    # Import library ExtraTreesClassifier
    from sklearn.ensemble import ExtraTreesClassifier
    # fit model ExtraTreesClassifier
    model = ExtraTreesClassifier()
    model.fit(X,y)


    print(model.feature_importances_)
    feat_importance = pd.Series(model.feature_importances_, index=X.columns)
    feat_importance.nlargest(10).plot(kind='barh')
    plt.show()


    # mendapatkan korelasi di setiap fitur dalam dataset
    corrmat = df1.corr()
    top_corr_features = corrmat.index
    plt.figure(figsize=(20,20))
    # plot heatmap
    h = sns.heatmap(df1[top_corr_features].corr(),annot=True,cmap="RdYlGn")
    df4 = df1.drop(['jumlah_terdampak_rw','jumlah_luka_ringan'], axis=1)
    return definition

def detect_outliers(df4, x):
    df = pd.read_csv("data-kejadian-bencana-banjir-provinsi-dki-jakarta-2020-bulan-januari-maret_tes.csv")
    df1 = df.drop(['jumlah_meninggal','jumlah_hilang','jumlah_luka_berat','jumlah_tempat_pengungsian','nilai_kerugian','lama_genangan','jumlah_pengungsi_tertinggi'], axis=1)
    df4 = df1.drop(['jumlah_terdampak_rw','jumlah_luka_ringan'], axis=1)
    Q1 = df4[x].describe()['25%']
    Q3 = df4[x].describe()['75%']
    IQR = Q3-Q1
    return df4[(df4[x] < Q1-1.5*IQR) | (df4[x] > Q3+1.5*IQR)]

def flood_cluster(hrs):
    df = pd.read_csv("data-kejadian-bencana-banjir-provinsi-dki-jakarta-2020-bulan-januari-maret_tes.csv")
    df1 = df.drop(['jumlah_meninggal','jumlah_hilang','jumlah_luka_berat','jumlah_tempat_pengungsian','nilai_kerugian','lama_genangan','jumlah_pengungsi_tertinggi'], axis=1)
    df4 = df1.drop(['jumlah_terdampak_rw','jumlah_luka_ringan'], axis=1)


    df4 = df4.drop((df4[df4['ketinggian_air_max(cm)']>199]).index, axis=0)
    df4 = df4.drop((df4[df4['ketinggian_air_min(cm)']>69]).index, axis=0)
    df4 = df4.drop((df4[df4['jumlah_terdampak_jiwa']>8000]).index, axis=0)

    from sklearn.preprocessing import LabelEncoder
    label_encoder = LabelEncoder()
    df4['kota_administrasi'] = label_encoder.fit_transform(df4['kota_administrasi'])
    df4['kelurahan'] = label_encoder.fit_transform(df4['kelurahan'])
    df4['rw'] = label_encoder.fit_transform(df4['rw'])
    df4['tanggal_kejadian'] = label_encoder.fit_transform(df4['tanggal_kejadian'])

    df5= df4.drop(['jumlah_terdampak_kk','tanggal_kejadian'], axis=1)
    df6 = df5.drop(['kecamatan'], axis=1)
    
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    std_atr=scaler.fit_transform(df6)
    std_atr=pd.DataFrame(std_atr,columns=df6.columns)

    X = df4.iloc[:,[4,10]].values

    from sklearn.cluster import KMeans
    wcss = []
    for i in range(1, 11):
        kmeans = KMeans(n_clusters = i, init = 'k-means++', random_state = 42)
        kmeans.fit(X)
        wcss.append(kmeans.inertia_)
    
    from sklearn.metrics import silhouette_score
    km=KMeans(n_clusters=4, random_state=42)
    km.fit(std_atr)
    score = silhouette_score(std_atr,km.labels_)
    
    # Proses K-Means Clustering
    kmeans = KMeans(n_clusters = 4, init = 'k-means++', random_state = 42)
    y_kmeans = kmeans.fit_predict(X)
    
    model=KMeans(4, random_state=42)
    model.fit(std_atr)
    clt=model.labels_
    result=pd.Series(clt,name="Cluster")
    result=pd.DataFrame(result)
    
    df5['Cluster'] = result
    
    smry = df5.iloc[:,[1,9]]
    kec = df5['kecamatan']
    Klaster = df5['Cluster']
    
    return smry

