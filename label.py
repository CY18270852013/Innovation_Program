import numpy as np

# Load the data and labels
date = list(np.load(r"C:\Users\Chen Yong\Python codes\date.npz", allow_pickle=True)['data'])
label = np.load(r"C:\Users\Chen Yong\Python codes\label8.npy", allow_pickle=True)

t = True

while t:
    start = input("输入要修改的起始照片名称(...+.jpg)：")
    end = input("输入要修改的末尾照片名称：")
    
    if start == 'q' :
        t = False
    elif  end == 'q':
        t = False
    else:
        try:
            start_idx = date.index(start[:] + '.npz')
            end_idx = date.index(end[:] + '.npz')
            
            # Ensure the range is correct
            if start_idx > end_idx:
                start_idx, end_idx = end_idx, start_idx
            
            change = input("修改后的标记 (白有云:[1,0,0] 黑:[0,1,0] 白无云:[0,0,1]): ")
            change_list = eval(change)
            
            # Convert the change list to a NumPy array of appropriate shape
            change_array = np.array(change_list)
            
            # Ensure the change_array matches the shape of individual label entries
            for i in range(start_idx, end_idx + 1):
                label[i] = change_array

        except ValueError as ve:
            print(f"Error: {ve}. 请确保输入的照片名称存在于数据中，并且标记格式正确。")
        except SyntaxError as se:
            print(f"Syntax Error: {se}. 请确保输入的标记格式正确。")

# Save the modified labels
np.save(r"C:\Users\Chen Yong\Python codes\label9.npy", label)
