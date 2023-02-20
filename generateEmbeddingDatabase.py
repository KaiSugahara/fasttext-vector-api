import sys
import os
import pandas as pd
import fasttext
import sqlite3

if __name__=="__main__":
    
    bin_path = sys.argv[1]
    
    """
        モデルの存在判定
    """
    if not os.path.exists(bin_path):
        
        raise Exception("指定のモデルが存在しません")
        
    
    """
        学習済みモデルのLoad
    """

    model = fasttext.load_model(bin_path)
    
    """
        Embedding Matrix
    """

    input_matrix = model.get_input_matrix()
    
    """
        Word -> Embedding
    """

    # Word -> ID
    word_map = pd.Series(range(len(model.get_words())), index=model.get_words())

    # Word -> Embedding
    word_embed = pd.DataFrame(input_matrix[word_map.values], index=word_map.index)
    word_embed.index.name = "WORD"
    word_embed.columns = "DIM_" + word_embed.columns.astype(str)

    # データベースに書き出し
    with sqlite3.connect('EMBEDDINGS.db') as con:

        word_embed.to_sql(
            name = 'WORD_EMBED',
            con = con,
            if_exists='replace', 
            index = True,
            method = 'multi',
            chunksize = 1000,
        )

    del(word_map, word_embed)
    
    """
        SubWord -> Embedding
    """

    # SubWord -> ID
    subwords = {}
    for word in model.get_words():
        subwords.update(dict(zip(*model.get_subwords(word))))
    subword_map = pd.Series(subwords)

    # SubWord -> Embedding
    subword_embed = pd.DataFrame(input_matrix[subword_map.values], index=subword_map.index)
    subword_embed.index.name = "SUBWORD"
    subword_embed.columns = "DIM_" + subword_embed.columns.astype(str)

    # データベースに書き出し
    with sqlite3.connect('EMBEDDINGS.db') as con:

        subword_embed.to_sql(
            name = 'SUBWORD_EMBED',
            con = con,
            if_exists='replace', 
            index = True,
            method = 'multi',
            chunksize = 1000,
        )

    del(subword_map, subword_embed)