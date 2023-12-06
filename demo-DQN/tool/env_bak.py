import pandas as pd
import numpy as np
from tool.fun import merge_time_intervals
from datetime import timedelta, datetime
from collections import Counter
from tool.dba import Mysql

DAY_TOTAL_SECONDS = 24 * 60 * 60
SHIFT_NAME = ['白班', '中班', '晚班']


def masked_argmin(x, condition):
    valid_idx = np.where(condition)[0]
    return valid_idx[x[valid_idx].argmin()]


def masked_argmax(x, condition):
    valid_idx = np.where(condition)[0]
    return valid_idx[x[valid_idx].argmax()]


class Database:
    def __init__(self, data):
        self.data = data
        self.plan_id = data.get('plan_id', 1)
        self.factory_id = data.get('factory_id', 1)
        self.create_user = data.get("create_user", "admin")
        self.all_date = list()
        self.all_machine = set()
        self.all_qad_code = set()
        self.delivery = self.get_delivery()
        self.ct = self.update_ct()
        self.start_time = {machine: None for machine in self.all_machine}
        self.break_time = {machine: [] for machine in self.all_machine}
        self.merge_table = self.update_machine()
        replace_start_time = Counter([x for x in self.start_time.values() if x is not None]).most_common(1)[0][0]
        for key in self.start_time:
            if self.start_time[key] is None:
                self.start_time[key] = replace_start_time
        self.job = self.update_job()

    def get_delivery(self):
        sql = f"""SELECT child_qad_code as qad_code, delivery_date, sum(planned_quantity * use_quantity) as demand_number
                FROM shift_scheduling a
                         INNER JOIN bom b on a.qad_code = b.parent_qad_code and a.factory_id = b.factory_id
                WHERE a.factory_id = '{self.factory_id}' and a.is_delete = 0 and b.is_delete=0
                  and plan_id = '{self.plan_id}' 
                  and b.child_qad_code is not null
                GROUP BY child_qad_code, delivery_date
                 """
        df_delivery = Mysql().read(sql)
        start_date = df_delivery['delivery_date'].min()
        end_date = start_date + timedelta(days=14)
        df_delivery = df_delivery[df_delivery["delivery_date"] < end_date]
        df_delivery['delivery_date'] = df_delivery['delivery_date'].apply(lambda x: x.strftime('%Y-%m-%d'))
        self.all_date = [x.strftime('%Y-%m-%d') for x in pd.date_range(start_date, end_date)]
        self.all_qad_code = list(set(df_delivery['qad_code']))
        delivery = {qad_code: {delivery_date: 0 for delivery_date in self.all_date} for qad_code in self.all_qad_code}
        for idx, row in df_delivery.iterrows():
            delivery[row["qad_code"]][row["delivery_date"]] = row['demand_number']
        return delivery

    def update_ct(self):
        sql = f"""
           SELECT qad_code, process_order, process_name, machine_name, ct FROM process_data
           where factory_id = {self.factory_id} and is_delete = 0 and process_name!='总成装配'
           order by qad_code, process_order"""
        df = Mysql().read(sql)
        ct = {}
        for (qad_code, process_order, process_name), g_data in df.groupby(['qad_code', 'process_order', 'process_name']):
            if qad_code not in ct:
                ct[qad_code] = []
            else:
                if np.max(g_data['ct']) == 0:
                    continue
                ct[qad_code].append({
                    "process_name": process_name,
                    "ct": {machine_name: ct / 60 / 60 / 24 for machine_name, ct in zip(g_data['machine_name'], g_data['ct'])}
                })
                if process_name != '总成装配':
                    for machine_name in g_data['machine_name']:
                        self.all_machine.add(machine_name)
        return ct

    def update_st_et(self, machine, date, start_time, end_time):
        st = (pd.to_datetime(date + ' ' + start_time + ':00') - self.start_time[machine]).total_seconds() / DAY_TOTAL_SECONDS
        et = (pd.to_datetime(date + ' ' + end_time + ':00') - self.start_time[machine]).total_seconds() / DAY_TOTAL_SECONDS
        if start_time < self.start_time[machine].strftime('%H:%M'):
            st += 1
        if end_time <= self.start_time[machine].strftime('%H:%M'):
            et += 1
        return [st, et]

    def add_rest_time(self):
        merge_table = []
        for machine in set(self.data['machine_schedule'].keys()).intersection(self.all_machine):
            data = self.data['machine_schedule'][machine]
            self.start_time[machine] = pd.to_datetime(datetime.today().strftime("%Y-%m-%d") + ' ' + data['machine_schedule_list'][0]['time_schedule_list'][0]['start_time'],
                                                      format='%Y-%m-%d %H:%M') + timedelta(days=1)
            for row in data['machine_schedule_list']:
                date_range = set(pd.date_range(start=row['start_date'], end=row['end_date']).strftime('%Y-%m-%d')).intersection(self.all_date)
                for date in date_range:
                    # 构建 [日期, 机器, 小时, 时段, 班次] 表用于结果构造
                    for hour in range(24):
                        shift_id = int(hour / (24 / len(row['time_schedule_list'])))
                        shift_name = SHIFT_NAME[row['time_schedule_list'][shift_id]['shift_order'] - 1]
                        period = [(timedelta(hours=hour) + self.start_time[machine]).strftime('%H:%M'),
                                  (timedelta(hours=hour + 1) + self.start_time[machine]).strftime('%H:%M')]
                        merge_table.append([hour, '~'.join(period), shift_name])
                    # 将休息时间表加入break_time
                    for rest_row in row['time_rest_list']:
                        self.break_time[machine].append(self.update_st_et(machine, date, rest_row['start_time'], rest_row['end_time']))
                    # 将节假日 固定时段休息加入break_time
                    for rest_row in data['time_festival_list']:
                        if rest_row['time_type'] == 2:
                            if pd.to_datetime(date).dayofweek + 1 == rest_row['day_of_week']:
                                self.break_time[machine].append(self.update_st_et(machine, date, rest_row['start_time'], rest_row['end_time']))
                        else:
                            st = (pd.to_datetime(rest_row['start_datetime'] + ':00') - self.start_time[machine]).total_seconds() / DAY_TOTAL_SECONDS
                            et = (pd.to_datetime(rest_row['end_datetime'] + ':00') - self.start_time[machine]).total_seconds() / DAY_TOTAL_SECONDS
                            self.break_time[machine].append([st, et])

        return pd.DataFrame(merge_table, columns=['hour', 'period', 'shift_name']).drop_duplicates(subset='hour')

    def add_occupancy(self):
        sql = f"""
        select * from (
            select factory_id, machine_name, start_time, end_time from maintenance_plan
            UNION
            select factory_id, machine_name, start_time, end_time from equipment_failure
            UNION
            select factory_id, machine_name, start_time, end_time from capacity_params
                          ) as mpef
        where factory_id = {self.factory_id} and end_time > sysdate() """
        df = Mysql().read(sql)
        for idx, row in df.iterrows():
            machine_name = row['machine_name']
            if machine_name not in self.all_machine:
                continue
            st = (pd.to_datetime(row['start_time']) - self.start_time[machine_name]).total_seconds() / DAY_TOTAL_SECONDS
            et = (pd.to_datetime(row['end_time']) - self.start_time[machine_name]).total_seconds() / DAY_TOTAL_SECONDS
            self.break_time[machine_name].append([st, et])

    def update_machine(self):
        merge_table = self.add_rest_time()
        self.add_occupancy()
        for machine in self.break_time:
            self.break_time[machine] = merge_time_intervals(self.break_time[machine])
        return merge_table

    def update_job(self):
        sql = f"""
        select p.qad_code, epq, packing_specification, safety_stock, inventory_quantity FROM (
        SELECT child_qad_code as qad_code, epq, packing_specification, safety_stock
        FROM product_data a
        INNER JOIN bom b on a.qad_code = b.parent_qad_code and a.factory_id = b.factory_id
        WHERE a.factory_id = {self.factory_id}
        union select qad_code, epq, packing_specification, safety_stock from product_data c
        WHERE c.factory_id = {self.factory_id}
                      ) p
        left join finished_goods_inventory d on p.qad_code = d.qad_code and d.factory_id = {self.factory_id}
            """
        product_data = Mysql().read(sql).fillna(0).drop_duplicates(subset='qad_code')
        product_data = product_data.set_index('qad_code').to_dict(orient='index')
        job = []
        for qad_code in self.delivery:
            if qad_code not in self.ct:
                continue
            if qad_code in product_data:
                inventory_quantity = product_data[qad_code]['inventory_quantity']
            else:
                inventory_quantity = 0
            for delivery_date in np.sort(list(self.delivery[qad_code].keys())):
                if qad_code in product_data:
                    demand = self.delivery[qad_code][delivery_date]
                    packing_specification = product_data[qad_code]['packing_specification']
                    epq = product_data[qad_code]['epq']
                    if demand >= 1:
                        safety_stock = product_data[qad_code]['safety_stock']
                        epq_num = np.ceil((float(demand) + safety_stock - inventory_quantity) / epq)
                        planned_quantity = packing_specification * packing_specification
                        inventory_quantity = inventory_quantity + planned_quantity - float(demand)
                        for i in range(int(epq_num)):
                            job.append({"delivery_date": delivery_date, "qad_code": qad_code, "packing_specification": packing_specification, "epq": epq,
                                        "planned_quantity": epq})
                else:
                    break
        job = pd.DataFrame(job)
        return job


class Machine:
    def __init__(self):
        self.logs = []
        self.break_time = []
        self.st = 0
        self.et = 0
        self.pt = 0
        self.oee = 0

    def add(self, st, et, qad_code, planned_quantity):
        self.logs.append([st, et, qad_code, planned_quantity])
        self.logs = sorted(self.logs)
        self.st = min(st, self.st)
        self.et = max(et, self.et)
        self.pt += et - st
        self.oee = self.pt / (self.et - self.st + 1)


def get_change_time(last_qad_code, next_qad_code):
    if next_qad_code == last_qad_code:
        return 0
    elif last_qad_code is None:
        return 0
    else:
        return 30 / 60 / 24


class JSP(Database):
    def __init__(self, data):
        super().__init__(data)
        # np.random.shuffle(self.job['planned_quantity'])
        # np.random.shuffle(self.job['delivery_date'])
        for var in ['planned_quantity', 'delivery_date']:
            x = self.job[var].values
            np.random.shuffle(x)
            self.job[var] = x
        self.machine = {machine: Machine() for machine in self.all_machine}
        self.maximum_op = np.array([len(self.ct[qad_code]) - 1 for qad_code in self.job.qad_code])
        self.current_op = np.zeros(len(self.job), dtype=int)
        self.arrive_time = np.array([self.all_date.index(delivery_date) for delivery_date in self.job.delivery_date], dtype=float)
        # 期望交付时间为到达时间后三天
        self.expected_time = self.arrive_time + 3
        self.start_time = self.arrive_time
        self.doing = np.zeros(len(self.job), dtype=int)
        self.total_op_remain_time = np.zeros(len(self.job), dtype=float)
        self.current_op_remain_time = np.zeros(len(self.job), dtype=float)
        self.current_machine = np.zeros(len(self.job), dtype=object)
        self.time_step = 0
        self.reset_env()
        self.r = self.expected_time - np.maximum(self.start_time, self.time_step) - self.total_op_remain_time
        self.state = self.get_state()
        self.logs = []

    def reset_env(self):
        for job_id, row in self.job.iterrows():
            current_op = self.current_op[job_id]
            ct = self.ct[row['qad_code']][current_op]['ct']
            machine_start_time = min(self.machine[machine_name].et for machine_name in ct)
            self.start_time[job_id] = max(machine_start_time, self.time_step, self.arrive_time[job_id], self.start_time[job_id])
            total_ct = 0
            for x in self.ct[row['qad_code']][current_op + 1:]:
                total_ct += np.mean(list(x['ct'].values()))
            self.total_op_remain_time[job_id] = total_ct * row['planned_quantity']
            machine_name = list(ct.keys())
            machine_ct = list(ct.values())
            self.current_op_remain_time[job_id] = np.min(machine_ct) * row['planned_quantity']
            self.current_machine[job_id] = machine_name[np.argmin(machine_ct)]
            self.total_op_remain_time[job_id] += self.current_op_remain_time[job_id]

    def update_job_time(self, job_id, machine_name):
        for idx in np.where(self.current_machine == machine_name)[0]:
            if idx != job_id:
                if self.doing[job_id]:
                    continue
                current_op = self.current_op[job_id]
                if current_op == self.maximum_op[job_id]:
                    continue
                row = self.job.iloc[idx]
                ct = self.ct[row['qad_code']][current_op]['ct']
                machine_start_time = min(self.machine[machine_name].et for machine_name in ct)
                self.start_time[idx] = max(machine_start_time, self.time_step, self.arrive_time[idx], self.start_time[idx])
                total_ct = 0
                for x in self.ct[row['qad_code']][current_op + 1:]:
                    total_ct += np.mean(list(x['ct'].values()))
                self.total_op_remain_time[idx] = total_ct * row['planned_quantity']
                machine_name = list(ct.keys())
                machine_ct = list(ct.values())
                self.current_op_remain_time[idx] = np.min(machine_ct) * row['planned_quantity']
                self.current_machine[idx] = machine_name[np.argmin(machine_ct)]

    def get_state(self):
        f1 = 1 - np.sum(self.current_op) / np.sum(self.maximum_op)
        f2 = np.mean([self.machine[machine].oee for machine in self.machine])
        f3 = np.min(self.start_time) / (np.max(self.start_time) + 1)
        f4 = np.max(self.start_time) / (np.min(self.start_time) + 1)
        f5 = np.min(self.total_op_remain_time) / (np.max(self.total_op_remain_time) + 1)
        f6 = np.max(self.total_op_remain_time) / (np.min(self.total_op_remain_time) + 1)
        edd = self.expected_time - self.total_op_remain_time - self.time_step
        f7 = np.min(edd) / (np.max(edd) + 1)
        f8 = np.max(edd) / (np.min(edd) + 1)
        return [f1, f2, f3, f4, f5, f6, f7, f8]

    def fifo(self, condition):
        return masked_argmin(self.arrive_time, condition)

    def spt(self, condition):
        return masked_argmin(self.current_op_remain_time, condition)

    def lwr(self, condition):
        return masked_argmin(self.total_op_remain_time, condition)

    def edd(self, condition):
        return masked_argmin(self.expected_time, condition)

    def random(self, condition):
        return masked_argmax(np.random.random(len(self.job)), condition)

    def update_doing(self):
        for x in filter(lambda x: (x[2] <= self.time_step) & (x[3] > self.time_step), self.logs):
            self.doing[x[0]] = 1
            self.current_op_remain_time[x[0]] = x[3] - self.time_step
            self.total_op_remain_time[x[0]] = self.total_op_remain_time[x[0]] + x[3] - self.time_step

    def reward(self, job_id):
        r = self.expected_time - np.maximum(self.start_time, self.time_step) - self.total_op_remain_time
        reward = (r[job_id] - self.r[job_id]) / np.sum(self.maximum_op + 1) * 100
        self.r = r
        return reward

    def step(self, action):
        condition = (self.current_op < self.maximum_op)
        if action == 0:
            job_id = self.fifo(condition)
        elif action == 1:
            job_id = self.spt(condition)
        elif action == 2:
            job_id = self.lwr(condition)
        elif action == 3:
            job_id = self.edd(condition)
        else:
            job_id = self.random(condition)
        row = self.job.iloc[job_id]
        current_op = self.current_op[job_id]
        ct = self.ct[row['qad_code']][current_op]['ct']
        st_ = [max(self.machine[machine_name].et, self.time_step, self.arrive_time[job_id], self.start_time[job_id]) for machine_name in ct]
        machine_id = np.argmin(st_)
        if action > 3:
            machine_id = np.random.choice(range(len(ct)))
        machine_name = list(ct.keys())[machine_id]
        st = st_[machine_id]
        et = st + ct[machine_name] * row['planned_quantity']
        self.logs.append([job_id, machine_name, st, et])
        self.machine[machine_name].add(st, et, row['qad_code'], row['planned_quantity'])
        self.current_op[job_id] += 1
        self.current_machine[job_id] = machine_name
        self.start_time[job_id] = et
        self.arrive_time[job_id] = et
        self.update_doing()
        self.update_job_time(job_id, machine_name)
        self.state = self.get_state()
        if np.sum(self.current_op < self.maximum_op) == 0:
            done = 1
        else:
            done = 0
        return self.state, self.reward(job_id), done
