import numpy as np
import pandas as pd
import sqlite3

def get_subwords(word):
    
    """
        func: 
            - 単語のサブワードを抽出（fastTextの処理に従う）
        args:
            - word: 対象の単語 str
        returns:
            - word: 対象の単語 str
            - subwords: 抽出されたサブワードのリスト list[]
    """
    
    _word = f"<{word}>"
    
    subwords = []
    
    for l in [6, 5, 4, 3]:
        for i in range(len(_word) - l + 1):
            subwords.append(_word[i:i+l])
            
    return word, subwords

def get_word_embedding(word, con):
    
    """
        func: 
            - 単語の埋め込みをデータベースから検索，検索結果を返す
            - 見つからない場合は空ベクトルを返す
        args:
            - word: 対象の単語 str
            - con: SQLのコネクタ
        returns:
            - 埋め込みベクトル np.array[1, embed_dim]
    """

    query = f"SELECT * FROM WORD_EMBED WHERE WORD = '{word}'"
    return pd.read_sql_query(query, con).iloc[:, 1:].values

def get_subword_embeddings(subwords, con):
    
    """
        func: 
            - サブワードの埋め込みをデータベースから検索，検索結果を返す
            - 見つからない場合は空ベクトルを返す
        args:
            - subwords: 対象のサブワードのリスト list[]
            - con: SQLのコネクタ
        returns:
            - 埋め込み行列 np.array[subword_size, embed_dim]
    """

    query = f"SELECT * FROM SUBWORD_EMBED WHERE SUBWORD IN (%s)" % ",".join(["?"]*len(subwords))
    return pd.read_sql_query(query, con, params=tuple(subwords)).iloc[:, 1:].values


from flask import Flask, jsonify, request
app = Flask(__name__)

@app.route("/", methods=["POST"])
def index():
    
    con = sqlite3.connect('EMBEDDINGS.db')

    word_list = request.json
    word_subwords_list = [get_subwords(word) for word in word_list]

    word_embeddings = {word: (get_word_embedding(word, con=con), get_subword_embeddings(subwords, con=con)) for word, subwords in word_subwords_list}

    word_embeddings = {word: np.vstack(embeds).mean(axis=0).tolist() for word, embeds in word_embeddings.items()}
    
    return jsonify(word_embeddings)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=80)