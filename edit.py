from datetime import datetime
dt  = datetime.now()
print(dt)
dt2 = datetime(2023, 1, 1, 14, 2)
days = (dt.timestamp() - dt2.timestamp()) // (24 * 60 * 60)
print(int(days))