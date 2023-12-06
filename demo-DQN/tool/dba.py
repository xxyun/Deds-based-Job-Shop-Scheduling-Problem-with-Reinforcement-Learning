import pandas as pd
import pymysql


class Mysql:
    def __init__(self):
        self.conn = pymysql.connect(host="10.90.15.233", port=3306, user='aps', passwd='Xghy2023@123', db='aps', charset='utf8')
        self.cursor = self.conn.cursor()

    def read(self, sql):
        self.cursor.execute(sql)
        var_name = [var[0] for var in self.cursor.description]
        df = self.cursor.fetchall()
        df = pd.DataFrame(list(df), columns=var_name)
        return df

    def execute(self, sql):
        self.cursor.execute(sql)
        self.conn.commit()

    def __del__(self):
        self.cursor.close()
        self.conn.close()
