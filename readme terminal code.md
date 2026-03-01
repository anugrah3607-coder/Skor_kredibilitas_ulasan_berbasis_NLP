
cd "E:\Essay statistic\model ai baru\review_credibility_nlp"
# virtual environtment 
.\.venv\Scripts\Activate.ps1
# cek kesiapan
python -c "import pandas, sklearn, scipy; print('env OK')"
# train data dari dataset 
python train.py --data examples\train_all.csv --out models\model_v2_final --n-splits 5
# cek sample banyak dari dataset
python predict.py --model-dir models\model_v2_final --input "data bersih\uji_2000_ulasan.csv" --output scored.csv
# prediksi satu ulasan 
python -c "from src.model import load_model, score_reviews; m=load_model('models/model_v1'); text='ULASAN TULIS; print(score_reviews(m,[text]).to_string(index=False))"

python predict.py --model-dir models\model_all_v1 --input "archive/unalabled_deception_opinion.csv" --output scoredbaru.csv
# output data mean
Statistik dasar (min/mean/max p_fake + deskripsi skor)
python -c "import pandas as pd; df=pd.read_csv('scoredbaru.csv'); print('p_fake min/mean/max:', df['p_fake'].min(), df['p_fake'].mean(), df['p_fake'].max()); print(df['credibility_score'].describe())"
# rate di bawah 30 
python -c "import pandas as pd; df=pd.read_csv('scoredbaru.csv'); print('audit_rate(score<30)=', (df['credibility_score']<30).mean())"
# evaluasi 5-fold 
python train.py --data examples\train_all.csv --out models\model_eval_v1 --n-splits 5
#