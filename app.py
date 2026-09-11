import streamlit as st
import datetime
import logging
import os
from typing import Dict, List, Tuple
import imagehash
from PIL import Image # 建議安裝 Pillow 套件來處理影像
import time
from ldap3 import Server, Connection, Tls
import ssl
import cv2
import numpy as np
from skimage.metrics import structural_similarity as ssim
import random
import string
from captcha.image import ImageCaptcha
from io import BytesIO
import io
import zipfile
from pypdf import PdfReader
# ─── 引入剛剛獨立出來的資料庫模組 ───
import db_helper
# 確保在 App 啟動時先初始化資料庫（建立檔案與資料表）
db_helper.init_db()


# 1. 初始化 Session State 變數
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False
if "captcha_text" not in st.session_state:
    st.session_state.captcha_text = ""

PHOTO_EXTENSIONS = ('.jpg', '.jpeg', '.png', '.gif', '.bmp')
COMPARISON_THRESHOLD = 15

# 1. 配置日誌設定 (維持不變)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s', # 移除固定 IP 字樣，讓 message 更彈性
    handlers=[
        logging.FileHandler("server_access.log", encoding='utf-8'),
        logging.StreamHandler()
    ]
)

# --- 1. 安全地載入網域對應表 ---
try:
    # 這行會把 secrets.toml 裡的 [AD_DOMAINS] 讀取成一個 dict
    DOMAIN_DICT = st.secrets["AD_DOMAINS"]
except KeyError:
    st.error("系統設定錯誤：找不到網域配置 (AD_DOMAINS)，請聯絡管理員。")
    st.stop()

def loop_folder(folder_path):
    photos = []
    for filename in os.listdir(folder_path):
        if filename.lower().endswith(PHOTO_EXTENSIONS):
            photos.append(filename)
    return photos

# 2. 產生隨機驗證碼文字與圖片的函式
def generate_captcha():
    # 隨機產生 4 位數的大寫字母與數字組合
    chars = string.ascii_uppercase + string.digits
    captcha_text = "".join(random.choice(chars) for _ in range(4))
    st.session_state.captcha_text = captcha_text
    
    # 使用 captcha 套件產生圖片
    image = ImageCaptcha(width=280, height=90)
    data = image.generate(captcha_text)
    return Image.open(BytesIO(data.read()))
                      
def load_and_hash_photos_8value(folder_path):
    valid_extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.webp')
    photos_hash_dict = {}
    
    all_files = os.listdir(folder_path)
    
    for file_name in all_files:
        if not file_name.lower().endswith(valid_extensions):
            continue
            
        file_path = os.path.join(folder_path, file_name)
        
        try:
            # ❌ 原本失敗的寫法：img_org = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
            
            # 💡 修正後的安全中文路徑讀取法：
            # 1. 使用 numpy 以二進位（uint8）讀取檔案內容
            img_array = np.fromfile(file_path, dtype=np.uint8)
            # 2. 將二進位數據解碼成 OpenCV 影像格式（先解成彩色 IMREAD_COLOR）
            img_bgr = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
            
            if img_bgr is None:
                print(f"無法讀取圖片: {file_name}")
                continue
                
            # 3. 再手動轉為灰階，符合後續 pHash 計算所需
            img_org = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
            
            # --- 以下維持你原本的幾何變換與 hash 計算邏輯 ---
            img_90  = cv2.rotate(img_org, cv2.ROTATE_90_CLOCKWISE)
            img_180 = cv2.rotate(img_org, cv2.ROTATE_180)
            img_270 = cv2.rotate(img_org, cv2.ROTATE_90_COUNTERCLOCKWISE)
            
            img_m   = cv2.flip(img_org, 1) 
            img_m90 = cv2.rotate(img_m, cv2.ROTATE_90_CLOCKWISE)
            img_m180= cv2.rotate(img_m, cv2.ROTATE_180)
            img_m270= cv2.rotate(img_m, cv2.ROTATE_90_COUNTERCLOCKWISE)
            
            images_list = [img_org, img_90, img_180, img_270, img_m, img_m90, img_m180, img_m270]
            hashes = []
            
            for img in images_list:
                pil_img = Image.fromarray(img)
                h = imagehash.phash(pil_img)
                hashes.append(h)
                
            photos_hash_dict[file_name] = hashes
            print(f"已成功計算 hashes: {file_name}")
            
        except Exception as e:
            print(f"處理檔案 {file_name} 時發生錯誤: {e}")
            
    return photos_hash_dict
def find_similar_photos_8value(photos_hash_dict, threshold):
    found_similarities = []
    file_names = list(photos_hash_dict.keys())
    
    # 倆倆不重複比對 (巢狀迴圈)
    for i in range(len(file_names)):
        for j in range(i + 1, len(file_names)):
            name1 = file_names[i]
            name2 = file_names[j]
            
            hash1_org = photos_hash_dict[name1][0] # 拿照片1的「原始狀態」
            hash2_variants = photos_hash_dict[name2] # 拿照片2的「8種變換狀態」
            
            # 計算 8 次，找最小的距離
            min_distance = min([hash1_org - h2 for h2 in hash2_variants])
            
            if min_distance <= threshold:
                # 找到了！記錄下來
                found_similarities.append((name1, name2, min_distance))
                
    return found_similarities
# 假設這是你的 compare 子程式 (可以放在檔案最上方或單獨的 module 匯入)
def compare_new(folder_path):
    print("starting compare process")
    print(folder_path)
    # photos_hash_dict = load_and_hash_photos(folder_path)
    photos_hash_dict = load_and_hash_photos_8value(folder_path)
    # print(photos_hash_dict)
    # found_similarities = find_similar_photos(photos_hash_dict, COMPARISON_THRESHOLD)
    found_similarities = find_similar_photos_8value(photos_hash_dict, COMPARISON_THRESHOLD)

    print("\n--- 相似照片對結果 (距離 <= {}) ---".format(COMPARISON_THRESHOLD))

    # 1. 取得目前時間並格式化 (年月日時分秒，例如: 20260522_111430)
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

    # 2. 定義目標資料夾路徑：reports/{使用者帳號}/
    # 💡 使用 os.path.join 可以自動處理不同作業系統的斜線方向 ( Windows 為 \ , Linux 為 / )
    user_id = st.session_state.get('username')
    report_dir = os.path.join("reports", user_id)

    # 3. 檢查資料夾是否存在，若不存在則自動建立 (exist_ok=True 代表資料夾已存在時不報錯)
    os.makedirs(report_dir, exist_ok=True)

    # 4. 定義完整檔案路徑 (包含資料夾與檔名)
    file_name = f"compare_result_{timestamp}.txt"
    full_file_path = os.path.join(report_dir, file_name)
    
    if found_similarities:
        # 先建立要寫入的文字內容
        output_text = f"--- 比對報告(小於15會被列出，距離0為完全相同，請依檔名檢視您上傳的照片) (執行時間: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}) ---\n"
        for name1, name2, distance in found_similarities:
            print(f"相似: {name1} vs {name2} | 距離: {distance}")
            output_text += f"相似: {name1} vs {name2} | 距離: {distance}\n"
    
        # 3. 寫入伺服器後台的 TXT 檔案（使用動態檔名）
        with open(full_file_path, "w", encoding="utf-8") as f:
            f.write("本系統係基於感知雜湊（pHash）演算法之「輔助比對工具」。\n")    
            f.write("系統所提供之相似度數值，僅供業務主管單位作為抽查與複核之決策參考，不具備法律上之最終判定效力。\n")
            f.write("-------------------\n")
            f.write(output_text) 
    else:
        print("沒有找到漢明距離小於閾值的照片對。")
        output_text = f"此次比對沒有找到漢明距離小於15的照片對\n"
        # 3. 寫入伺服器後台的 TXT 檔案（使用動態檔名）
        with open(full_file_path, "w", encoding="utf-8") as f:
            f.write("本系統係基於感知雜湊（pHash）演算法之「輔助比對工具」。\n")    
            f.write("系統所提供之相似度數值，僅供業務主管單位作為抽查與複核之決策參考，不具備法律上之最終判定效力。\n")
            f.write("-------------------\n")
            f.write(output_text)
    photos_name = loop_folder(folder_path)

    with open(full_file_path, "a", encoding="utf-8") as f:
        f.write("-------------------\n")
        f.write("--- 以下為參與此次比對所有照片檔名 ---\n")
        for name in photos_name:
            f.write(f"{name}\n")
    st.success(f"檔案已儲存為：`{file_name}` 請至比對結果查詢頁面下載")
# 2. 定義獲取 IP 的函式 (確保能抓到 Nginx 轉發後的 IP)
def get_client_ip():
    headers = st.context.headers
    
    # 1. 嘗試所有可能的 Nginx/Proxy 標頭 (不分大小寫)
    # 這裡檢查了常見的轉發標頭
    ip_keys = [
        "x-forwarded-for", "X-Forwarded-For", 
        "x-real-ip", "X-Real-Ip", 
        "remote-addr"
    ]
    
    for key in ip_keys:
        if key in headers:
            # 取得該標頭的值
            ip = headers.get(key)
            # 如果是 X-Forwarded-For，可能包含多個 IP (以逗號隔開)，取第一個
            return ip.split(",")[0].strip()
            
    # 2. 如果都沒有，但 Host 是 localhost，則標記為 127.0.0.1
    host = headers.get("host", "").lower()
    if "localhost" in host or "127.0.0.1" in host:
        return "127.0.0.1 (Local Server)"
        
    return "Unknown IP"

# 3. 執行紀錄邏輯
client_ip = get_client_ip()


def write_audit_log(user, action, status, detail=""):
    """
    業界常見的稽核日誌格式
    """
    logging.info(f"User: {user} | Action: {action} | Status: {status} | Detail: {detail}")



# 1. 頁面設定：確保 initial_sidebar_state 為 "expanded" (展開)
st.set_page_config(
    page_title="佐證照片比對系統", 
    layout="wide", 
    initial_sidebar_state="expanded" 
)

# 2. 精準隱藏：只隱藏選單和 Deploy，不隱藏整個 Header
hide_st_style = """
            <style>
            /* 隱藏右上角三條線選單 */
            #MainMenu {visibility: hidden;}
            
            /* 隱藏頂部裝飾橫條 */
            header {visibility: hidden;}
            
            /* 專門針對新版 Deploy 按鈕進行隱藏 */
            .stAppDeployButton {display: none !important;}
            
            /* 隱藏底部 "Made with Streamlit" */
            footer {visibility: hidden;}
            
            /* 調整側邊欄上方間距，避免因為隱藏 header 導致太靠上 */
            .stSidebar {padding-top: 2rem;}
            </style>
            """
st.markdown(hide_st_style, unsafe_allow_html=True)


def authenticate_ad_safe(username, password, domain):
    """
    對 AD 伺服器進行帳密驗證。
    - username: 來自前端輸入的帳號
    - password: 來自前端輸入的密碼
    """
    # 1. 安全載入靜態的伺服器基礎設施情報 (絕對不寫死在程式碼中)
    ad_server = st.secrets.get("AD_SERVER_HOST") or os.getenv("AD_SERVER_HOST")
    ad_port_env = st.secrets.get("AD_SERVER_PORT") or os.getenv("AD_SERVER_PORT", "636")
    
    # 阻擋機制：如果伺服器忘記設定環境變數，直接回報錯誤，避免程式崩潰
    if not ad_server:
        logging.error("[AD Auth] 系統啟動錯誤：找不到 AD_SERVER_HOST 設定。")
        return False

    try:
        ad_port = int(ad_port_env)
    except ValueError:
        ad_port = 636  # 預設保護機制

    # 2. 建立 TLS 設定
    # 註：若公司內部 AD 有發行正式憑證，建議未來將 CERT_NONE 改為 ssl.CERT_REQUIRED
    tls_config = Tls(validate=ssl.CERT_NONE)
    
    # 3. 建立 Server 物件 (動態帶入剛剛抓到的變數)
    server = Server(
        ad_server, 
        port=ad_port, 
        use_ssl=True, 
        tls=tls_config, 
        connect_timeout=5
    )
    
    try:
        user_principal = f"{domain}\\{username}" 
        
        # 5. 建立連線並嘗試綁定 (Bind)
        conn = Connection(
            server, 
            user=user_principal, 
            password=password, 
            authentication='SIMPLE'
        )
        
        if conn.bind():
            conn.unbind() # 驗證成功後養成好習慣立刻斷開
            return True
        return False
        
    except Exception as e:
        # 6. 資安防護：把 print 換成 logging
        # 絕對不要把 e 當作 return 傳給前端，避免駭客看到詳細的連線錯誤訊息
        logging.error(f"[AD Auth] 驗證過程發生異常 (User: {username}): {e}")
        return False


def cv_imread_chinese_path(file_path):
    
    # 使用 NumPy 和 cv2.imdecode() 來安全讀取包含中文路徑的圖片。
    
    try:
        # 1. 使用 Python 內建的 open() 函式以二進制模式 ('rb') 讀取整個檔案
        with open(file_path, 'rb') as f:
            # 將檔案內容讀取為一個位元組數組 (bytes array)
            binary_data = f.read()
        
        # 2. 將位元組數組轉換為 NumPy 陣列
        #    np.frombuffer() 從緩衝區創建一個陣列
        np_array = np.frombuffer(binary_data, np.uint8)
        
        # 3. 使用 cv2.imdecode() 從記憶體中的緩衝區解碼圖片
        #    cv2.IMREAD_COLOR 確保讀取為彩色圖片 (BGR 格式)
        img = cv2.imdecode(np_array, cv2.IMREAD_COLOR)
        
        if img is None:
            print(f"警告：成功讀取檔案但無法解碼，可能是圖片格式不受支援或檔案損壞。")
            
        return img
        
    except FileNotFoundError:
        print(f"錯誤：找不到檔案路徑：{file_path}")
        return None
    except Exception as e:
        print(f"讀取或解碼圖片時發生錯誤：{e}")
        return None
def sort_and_write_results(results_list: List, userfolder, output_filename: str = "comparison_results.txt"):
    """
    對比對結果列表進行排序，並將結果寫入 TXT 檔案。

    Args:
        results_list: 包含 (圖1檔名, 圖2檔名, 分數) 的列表。
        output_filename: 輸出檔案的名稱。
    """
    print(f"--- 開始排序 {len(results_list)} 筆資料 ---")
    
    # 1. 排序操作
    # key=lambda x: x[0] 指定以 tuple 的第一個元素 (圖1檔名) 進行排序
    # reverse=True 設置為降冪排序 (從 Z 到 A)
    sorted_results = sorted(
        results_list, 
        # key=lambda x: x[0], 
        key=lambda x: x[2], 
        reverse=True
    )
    
    print("排序完成：依圖1檔名降冪排序 (Z -> A)。")
    timestamp = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
    output_filename = f"comparison_results_{timestamp.replace(':', '-')}.txt"
    output_filename = os.path.join(userfolder, output_filename)
    # 2. 寫入 TXT 檔案
    try:
        with open(output_filename, 'w', encoding='utf-8') as f:
            # 寫入標題行
            f.write("本系統係基於感知雜湊（pHash）演算法之「輔助比對工具」。\n")
            f.write("系統所提供之相似度數值，僅供業務主管單位作為抽查與複核之決策參考，不具備法律上之最終判定效力。\n")
            f.write("圖1檔名,圖2檔名,比對後相似百分比(%)\n")
            
            # 遍歷排序後的結果，並將每一行寫入檔案
            for filename1, filename2, score in sorted_results:
                # 將 tuple 格式化為 CSV 格式的字串
                rounded_score = round(float(score), 2)
                line = f"{filename1},{filename2},{rounded_score}\n"
                f.write(line)
        
        print(f"✅ 成功將結果寫入檔案: {output_filename}")
        
    except IOError as e:
        print(f"❌ 檔案寫入錯誤: {e}")




def find_similar_photos(photo_hashes: Dict[str, imagehash.ImageHash], 
                       threshold: int = COMPARISON_THRESHOLD) -> List[Tuple[str, str, int]]:
    """
    對字典中所有照片的 pHash 值進行兩兩比對。

    Args:
        photo_hashes: 包含 {檔名: pHash 值} 的字典。
        threshold: 漢明距離的閾值。距離小於或等於此值的照片將被視為相似。

    Returns:
        一個包含相似照片對結果的列表，格式為 [(檔名A, 檔名B, 漢明距離), ...]
    """
    
    # 將字典的鍵 (檔名) 和值 (Hash) 分別轉換為列表
    # 這樣可以方便地使用索引 i 和 j 進行迭代
    filenames = list(photo_hashes.keys())
    hashes = list(photo_hashes.values())
    
    num_photos = len(filenames)
    similar_pairs = []
    
    print(f"--- 開始兩兩比對 {num_photos} 張照片 ({num_photos * (num_photos - 1) // 2} 次比較) ---")
    start_time = time.time()

    # 使用嵌套迴圈進行兩兩比對
    # 外層迴圈 i 從 0 到 num_photos - 2
    for i in range(num_photos):
        
        # 內層迴圈 j 從 i + 1 開始，確保：
        # 1. 避免重複比較 (A vs B 後不比較 B vs A)
        # 2. 避免與自身比較 (i != j)
        for j in range(i + 1, num_photos):
            
            # 取出檔名和 Hash 值
            filename_i = filenames[i]
            hash_i = hashes[i]
            
            filename_j = filenames[j]
            hash_j = hashes[j]
            
            # 這是純 CPU 運算，速度極快
            distance = hash_i - hash_j
            
            # 檢查漢明距離是否小於閾值
            if distance <= threshold:
                similar_pairs.append((filename_i, filename_j, distance))
                
        # 簡單的進度輸出 (每處理 100 張照片輸出一次)
        if (i + 1) % 100 == 0:
            print(f"  > 已處理 {i + 1} / {num_photos} 張照片...")

    end_time = time.time()
    
    # 輸出計時結果
    print(f"--- 比對完成，總共耗時: {end_time - start_time:.4f} 秒 ---")
    
    return similar_pairs
def load_and_hash_photos(directory: str) -> Dict[str, imagehash.ImageHash]:
    """
    掃描指定目錄及其子目錄下的所有照片，計算並返回檔名和 pHash 值的字典。
    """
    photo_hashes: Dict[str, imagehash.ImageHash] = {}
    
    print(f"--- 開始掃描目錄並計算 pHash: {directory} ---")
    
    # 使用 os.walk 遞迴地遍歷目錄和子目錄
    for root, _, files in os.walk(directory):
        for filename in files:
            # 檢查檔案副檔名是否為照片
            if filename.lower().endswith(PHOTO_EXTENSIONS):
                # 構造完整的檔案路徑
                full_path = os.path.join(root, filename)
                
                try:
                    # 1. I/O 讀取操作 (這是耗時部分，但只需要執行一次)
                    img = Image.open(full_path)
                    
                    # 2. 計算 pHash
                    # 將檔名作為 key，pHash 物件作為 value
                    photo_hashes[filename] = imagehash.phash(img)
                    
                    # 釋放圖片物件，防止記憶體過度佔用 (特別是處理大量圖片時)
                    img.close() 
                    
                except Exception as e:
                    print(f"警告：無法處理檔案 {filename} ({e})")
                    continue

    print(f"--- pHash 計算完成，共處理 {len(photo_hashes)} 張照片 ---")
    return photo_hashes
def compare_images(imagePair, folder_path):
    BASE_DIR = os.path.join(os.getcwd(), folder_path)
    img1_file_name = imagePair[0]
    img2_file_name = imagePair[1]
    img1_path = os.path.join(BASE_DIR, img1_file_name)
    img2_path = os.path.join(BASE_DIR, img2_file_name)
    try:
        # Load images
        # img1 = cv2.imread(img1_path)
        # img2 = cv2.imread(img2_path)
        # load images support chinese character path
        img1 = cv_imread_chinese_path(img1_path)
        img2 = cv_imread_chinese_path(img2_path)
        
        # Convert to grayscale
        gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
        gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
        
        # Resize to same dimensions
        h = min(gray1.shape[0], gray2.shape[0])
        w = min(gray1.shape[1], gray2.shape[1])
        gray1 = cv2.resize(gray1, (w, h))
        gray2 = cv2.resize(gray2, (w, h))
        
        # Calculate SSIM (Structural Similarity Index)
        score, _ = ssim(gray1, gray2, full=True)
        percentage = score * 100
        
        # print(f"Similarity: {percentage:.2f}%")
        
        # self.result_label.config(text=f"Similarity: {percentage:.2f}%")
        if percentage >= 0.0:
            return (img1_file_name, img2_file_name, percentage)
        else:
            return None
    except Exception as e:
    # messagebox.showerror("Error", f"Failed to compare images: {str(e)}")
        print(f"Failed to compare images: {str(e)}")
def compare(folder_path):
    print("starting compare process")
    print(folder_path)
    # for file in os.listdir(folder_path):
        # print(file)

    photos_hash_dict = load_and_hash_photos(folder_path)
    # print(photos_hash_dict)
    found_similarities = find_similar_photos(photos_hash_dict, COMPARISON_THRESHOLD)
    print("\n--- 相似照片對結果 (距離 <= {}) ---".format(COMPARISON_THRESHOLD))
    if found_similarities:
        for name1, name2, distance in found_similarities:
            print(f"相似: {name1} vs {name2} | 距離: {distance}")
        print(str(len(found_similarities)) + "組hash相似照片對")
    else:
        print("沒有找到漢明距離小於閾值的照片對。")


    # version now
    # sort_and_write_results(found_similarities, folder_path)
    
    # version better
    # 以下對phash低於篩選值的照片對進行ssim分析
    result_list = []
    for pair in found_similarities:
        print(pair)
        result = compare_images((pair[0], pair[1]), folder_path)
        if result is not None:
            result_list.append(result)
    print(result_list)
    print(len(result_list))
    sort_and_write_results(result_list, folder_path)
def get_photo_count(username):
    user_folder = os.path.join("uploads", "photos", username)
    if not os.path.exists(user_folder):
        return 0
    valid_extensions = ('.jpg', '.jpeg', '.png')
    files = [f for f in os.listdir(user_folder) if f.lower().endswith(valid_extensions)]
    return len(files)



# --- 登入畫面 ---
def login_page():
    """佐證照片比對系統 - 登入畫面

    整合功能：
    - 連續打錯密碼 3 次鎖定 15 分鐘（以 SQLite 資料庫跨 Session 鎖定）
    - 不論成功、失敗、遭鎖定攔截，皆確實寫入 login_logs 流水帳
    - 網域與 LDAP/AD 驗證、圖形驗證碼防護
    """
    st.title("🔐 佐證照片比對系統登入")

    # 初始化驗證碼圖片
    if (
        "captcha_text" not in st.session_state
        or not st.session_state.captcha_text
    ):
        st.session_state.captcha_img = generate_captcha()

    # 建立登入表單
    with st.form("login_form", clear_on_submit=False):
        # 1. 網域與帳密輸入
        selected_domain_key = st.selectbox(
            "請選擇登入網域",
            options=list(DOMAIN_DICT.keys()),
        )
        username = st.text_input("帳號 (Windows ID)").strip()  # 自動去除前後空格
        password = st.text_input("密碼", type="password")

        # ─── 驗證碼區塊（緊湊排版） ───
        st.write("---")  # 加一條輕微的分隔線
        st.image(st.session_state.captcha_img, caption="驗證碼（不分大小寫）")

        # 點擊會立刻刷新表單重新產生驗證碼
        refresh_captcha = st.form_submit_button(
            "🔄 看不清？重新產生驗證碼", width='stretch'
        )

        user_captcha_input = st.text_input("請輸入上方驗證碼")
        st.write("---")

        # 2. 主要登入按鈕
        submit_button = st.form_submit_button(
            "確認登入", type="primary", width='stretch'
        )

    # ─── 後端邏輯判斷 ───

    # 優先判斷：如果是點擊「重新產生驗證碼」
    if refresh_captcha:
        st.session_state.captcha_img = generate_captcha()
        st.rerun()

    # 如果是點擊「確認登入」
    if submit_button:
        # 防呆：未填寫完整攔截
        if not username or not password or not user_captcha_input:
            st.warning("⚠️ 請完整填寫帳號、密碼與驗證碼")
            return

        # ─── 核心安控檢查：從 SQLite 檢查帳號是否處於 15 分鐘鎖定狀態 ───
        is_locked, remaining_time_str = db_helper.get_lockout_status(username)
        if is_locked:
            st.error(
                f"❌ 該帳號登入失敗次數過多，已被暫時鎖定！請等待 {remaining_time_str} 後再試。"
            )

            # 【軌跡留存】即便被阻擋，也寫入一筆 BLOCKED 歷史日誌
            db_helper.log_event(
                username,
                "BLOCKED",
                f"帳號鎖定中嘗試登入（剩餘：{remaining_time_str}）",
                client_ip,
            )
            logging.warning(
                f"[AUTH] {client_ip} - Blocked login attempt for LOCKED user '{username}'"
            )
            return

        # ─── 檢查一：驗證碼錯誤（不計入密碼錯誤次數，以免干擾使用者） ───
        if user_captcha_input.upper() != st.session_state.captcha_text.upper():
            st.error("❌ 驗證碼輸入錯誤，請重新輸入！")
            time.sleep(2)
            st.session_state.captcha_img = generate_captcha()
            st.rerun()

        # ─── 檢查二：LDAP/AD 帳密驗證 ───
        with st.spinner(f"正在連線至 {selected_domain_key} ..."):
            # 假設你在程式碼開頭已經宣告了 DOMAIN_DICT = st.secrets["AD_DOMAINS"]
            real_domain_value = DOMAIN_DICT.get(selected_domain_key)

            # 確保字典裡真的有這個選項，避免發生 KeyError
            if not real_domain_value:
                st.error("系統錯誤：無效的網域選項。")
                st.stop()

            if authenticate_ad_safe(
                username, password, real_domain_value
            ):

                # 【軌跡留存】1. 寫入 SUCCESS 歷史流水帳日誌
                db_helper.log_event(
                    username,
                    "SUCCESS",
                    f"透過網域 {selected_domain_key} 登入成功",
                    client_ip,
                )

                # 2. 登入成功，將該帳號的錯誤計數器重設歸零
                db_helper.reset_lockout(username)

                # 3. 寫入 Streamlit 全域變數以利後續頁面存取
                st.session_state["logged_in"] = True
                st.session_state["username"] = username
                st.session_state["domain"] = selected_domain_key

                logging.info(
                    f"[AUTH] {client_ip} - User '{username}' logged in successfully via Domain: {selected_domain_key}"
                )
                st.success("🎉 登入成功！")
                st.rerun()
            else:
                # 【軌跡留存】登入失敗：內部會將計數器 +1 並自動寫入 FAILED 歷史流水帳
                current_count = db_helper.record_failed_attempt(
                    username, client_ip
                )

                auth_result = "帳號密碼錯誤或網域連線逾時"
                logging.warning(
                    f"[AUTH] {client_ip} - Login FAILED: User '{username}' | Domain: {selected_domain_key} | Count: {current_count}/{db_helper.MAX_ATTEMPTS}"
                )

                # 根據剩餘次數給予使用者友善的資安警示
                if current_count >= db_helper.MAX_ATTEMPTS:
                    st.error(
                        f"❌ 驗證失敗。已連續錯誤 {current_count} 次，帳號將鎖定 {db_helper.LOCKOUT_DURATION} 分鐘！"
                    )
                else:
                    st.error(
                        f"❌ 驗證失敗：帳號密碼錯誤。再錯誤 {db_helper.MAX_ATTEMPTS - current_count} 次帳號將被鎖定。"
                    )

                time.sleep(2)
                st.session_state.captcha_img = generate_captcha()
                st.rerun()


# --- 管理員頁面 ---
def admin_dashboard():
    st.title("🚀 管理員後台")
    st.write(f"歡迎回來，管理者 {st.session_state['username']}")
    
    st.metric("今日伺服器負載", "20%", "-2%")
    st.subheader("使用者管理清單")
    st.table([{"ID": 1, "User": "general_user", "Status": "Active"}])
    
    if st.button("執行系統清理"):
        st.success("清理完成！")

# --- 一般使用者頁面 ---
def user_dashboard():
    # --- 1. 建立頂部導覽列 ---
    # 使用 8:2 的比例，讓標題佔大空間，按鈕縮在右邊
    col_title, col_logout = st.columns([8, 2])
    
    with col_title:
        st.title("🏠 佐證照片比對系統")
        st.write(f"您好，**{st.session_state.get('username', '使用者')}**，歡迎登入。")
        st.write("本系統僅用於照片比對，可能因系統負載定期清理資料，請妥善保管您的照片原始檔。")
        st.write("施工照片之真實性審查、現場查核以及最終結果之核定，其行政管理責任仍歸屬於【業務主辦/審核單位】。使用單位無法因本系統未標示風險，而免除其依合約及法規應盡之查驗與審查義務。")
    with col_logout:
        # 加一點間距讓按鈕對齊標題高度
        st.write("") 
        if st.button("🚪 登出系統", key="main_logout", width='stretch', type="secondary"):
            # 1. 取得資訊
            username = st.session_state.get('username', '未知使用者')

            # 2. 紀錄 Log (使用你指定的格式)
            logging.info(f"[AUTH] {client_ip} - User '{username}' logged out")
            
            # 3. 清除 Session 並重導向
            for key in list(st.session_state.keys()):
                del st.session_state[key]
            
            st.rerun()

    st.divider() # 畫一條橫線區隔導覽列與主內容
    # --- 在程式的最開頭或 user_dashboard 開始處 ---
    if "upload_success" not in st.session_state:
        st.session_state["upload_success"] = False
        st.session_state["compare_success"] = False

    # 建立分頁，讓介面更整齊
    tab1, tab2, tab5, tab6, tab7 = st.tabs(["📊 數據概覽", "📤 批次照片上傳", "🔍 啟動照片比對功能_new", "📋 比對結果查詢", "📷抽取PDF內照片"])
    
    with tab1:
        st.subheader("📊 數據概覽")
        
        # 1. 取得使用者資料夾路徑
        user_name = st.session_state['username']
        user_folder = os.path.join("uploads", "photos", user_name)
        
        # 2. 取得照片數量
        current_count = get_photo_count(st.session_state['username'])

        col1, col2 = st.columns(2)
        with col1:
            st.metric(label="已上傳照片總數", value=f"{current_count} 張")
        
        st.write("---")
        
        # 3. 取得所有照片檔名
        photo_files = []
        if os.path.exists(user_folder):
            valid_extensions = ('.jpg', '.jpeg', '.png')
            photo_files = [f for f in os.listdir(user_folder) if f.lower().endswith(valid_extensions)]
        
        # 4. 顯示縮圖區
        if photo_files:
            st.subheader("🖼️ 我的相簿縮圖")
            
            cols_per_row = 4
            # 將檔案列表分組為每 4 個一組的 row
            rows = [photo_files[i:i + cols_per_row] for i in range(0, len(photo_files), cols_per_row)]
            
            for row_files in rows:
                cols = st.columns(cols_per_row)
                for idx, file_name in enumerate(row_files):
                    img_path = os.path.join(user_folder, file_name)
                    
                    with cols[idx]:
                        try:
                            # 處理圖片縮圖
                            with Image.open(img_path) as img:
                                img.thumbnail((150, 150)) 
                                # 使用 2026 最新規範：width='stretch'
                                st.image(img, caption=file_name, width='stretch')
                            
                            # 刪除按鈕：同樣使用 width='stretch' 確保對齊
                            if st.button("🗑️ 刪除", key=f"del_{file_name}", type="secondary", width='stretch'):
                                os.remove(img_path)
                                st.success(f"已刪除 {file_name}")
                                st.rerun() # 立即刷新頁面反映變化

                        except Exception as e:
                            st.error("讀取失敗")
                            if st.button("🗑️ 強制刪除", key=f"err_del_{file_name}", width='stretch'):
                                if os.path.exists(img_path):
                                    os.remove(img_path)
                                st.rerun()
        else:
            st.warning("目前資料夾中沒有照片。")
    
    with tab2:
        st.subheader("📤 批次照片上傳")
        # 在頁面頂端或適當位置顯示持續的成功訊息
        if st.session_state["upload_success"]:
            st.success("🎉 所有照片已成功儲存至伺服器，您可以切換至數據概覽查看。")
            
        uploaded_files = st.file_uploader(
            "選擇照片進行預覽...", 
            type=["jpg", "png", "jpeg"], 
            accept_multiple_files=True,
            key="file_uploader" # 加入 key 方便後續判斷
        )

        # 邏輯：如果使用者「重新選擇」或「清空」了檔案，就重置成功狀態，讓按鈕變回可點擊
        if "previous_files" not in st.session_state:
            st.session_state["previous_files"] = []
        
        if uploaded_files != st.session_state["previous_files"]:
            st.session_state["upload_success"] = False
            st.session_state["previous_files"] = uploaded_files

        if uploaded_files:
            # 預覽區
            cols = st.columns(3)
            for index, uploaded_file in enumerate(uploaded_files):
                with cols[index % 3]:
                    img = Image.open(uploaded_file)
                    st.image(img, caption=uploaded_file.name, width="stretch")
            
            st.divider()

            # --- 動態按鈕設定 ---
            if st.session_state["upload_success"]:
                button_text = "✅ 照片已儲存"
                is_disabled = True
                st.session_state["upload_success"] = False
            else:
                button_text = "🚀 確定儲存並上傳"
                is_disabled = False

            # 顯示按鈕
            if st.button(button_text, width="stretch", type="primary", disabled=is_disabled):
                with st.spinner('儲存中...'):
                    user_name = st.session_state['username']
                    user_folder = os.path.join("uploads", "photos", user_name)
                    os.makedirs(user_folder, exist_ok=True)
                    
                    for uploaded_file in uploaded_files:
                        save_path = os.path.join(user_folder, uploaded_file.name)
                        with open(save_path, "wb") as f:
                            f.write(uploaded_file.getbuffer())
                    
                    # 更新狀態並觸發重新整理
                    st.session_state["upload_success"] = True
                    st.toast("照片上傳成功！", icon="🎉")
                    st.rerun()

    with tab6:
        st.subheader("📋 比對結果查詢")
        st.write("請自行下載文字檔檢視比對結果。")
        
        # 1. 取得使用者資料夾路徑
        user_name = st.session_state['username']
        user_folder = os.path.join("reports", user_name)
        # 這裡可以加入一個區塊來顯示比對結果，例如表格或列表
        # 1. 取得所有 .txt 結尾的檔案
        try:
            files = [f for f in os.listdir(user_folder) if f.endswith('.txt')]
        except FileNotFoundError:
            st.error(f"找不到資料夾：{user_folder}")
            files = []

        if files:
            files.sort(reverse=True) # 依修改時間排序，最新的在前
            for file_name in files:
                file_path = os.path.join(user_folder, file_name)
                
                # 使用三欄配置：檔名、下載、刪除
                col1, col2, col3 = st.columns([3, 1, 1])
                
                with col1:
                    st.text(f"📄 {file_name}")
                
                with col2:
                    with open(file_path, "rb") as f:
                        st.download_button(
                            label="下載",
                            data=f,
                            file_name=file_name,
                            mime="text/plain",
                            key=f"dl_{file_name}"
                        )
                
                with col3:
                    # 刪除按鈕
                    if st.button("刪除", key=f"del_{file_name}", type="secondary"):
                        try:
                            os.remove(file_path)
                            st.success(f"已刪除 {file_name}")
                            st.rerun() # 關鍵：刪除後強制重新整理頁面
                        except Exception as e:
                            st.error(f"刪除失敗: {e}")
                
                st.divider()
        else:
            st.info("資料夾內目前沒有 .txt 檔案。")
    # 主程式區域
    with tab5:
        st.subheader("🔍 啟動照片比對功能_new")
        # 💡 1. 檢查之前是否有留存「比對成功」的狀態，有的話就顯示它
        if st.session_state.get('compare_success', False):
            st.success("✨ 比對完成！報告已成功產生，請前往下載分頁。")
            # 顯示完後，立刻把狀態重設為 False，避免使用者切換其他按鈕時這行字一直卡在畫面上
            st.session_state['compare_success'] = False
        # 2. 建立按鈕
        if st.button("🚀 開始比對", type="primary"):
            with st.spinner("比對執行中，請稍候..."):
                try:
                    # 執行你的 compare 子程式
                    user_name = st.session_state['username']
                    user_folder = os.path.join("uploads", "photos", user_name)
                    
                    # 假設你的 compare_new 內部最後有成功寫入 txt 檔
                    compare_new(user_folder)
                    
                    # 💡 2. 不要當下顯示 success，而是改成在狀態機裡蓋章
                    st.session_state['compare_success'] = True
                    
                    # 💡 3. 重整頁面。重新跑一次後，會觸發最上方的 if 判斷，把 success 顯示出來！
                    st.rerun()
                    
                except Exception as e:
                    st.error(f"❌ 比對過程中發生錯誤: {e}")
    with tab7:
        st.subheader("📷 抽取 PDF 內照片")
        st.write("上傳您的 PDF 檔案，系統將自動解析並將所有內嵌的圖片打包成 ZIP 檔供您下載。（本功能採用商業合規的 pypdf 解析引擎）")

        # 1. 建立檔案上傳器 (加上 key 避免與其他 Tab 的 uploader 衝突)
        uploaded_pdf = st.file_uploader(
            "請選擇 PDF 檔案", 
            type=["pdf"], 
            key="pdf_image_extractor"
        )

        if uploaded_pdf is not None:
            # 2. 使用 spinner 讓使用者知道系統正在處理中
            with st.spinner("正在瘋狂解析 PDF 並提取圖片中，請稍候..."):
                try:
                    # 初始化 PdfReader
                    reader = PdfReader(uploaded_pdf)
                    extracted_images = []
                    
                    # 3. 逐頁解析 PDF 尋找圖片
                    for page_idx, page in enumerate(reader.pages):
                        # page.images 是一個包含 ImageFile 物件的列表
                        for img_obj in page.images:
                            # img_obj.name 是原本封裝的檔名，img_obj.data 是圖片的二進位資料
                            file_name = f"page_{page_idx + 1}_{img_obj.name}"
                            extracted_images.append((file_name, img_obj.data))

                    image_count = len(extracted_images)

                    # 4. 根據結果呈現不同的 UI 回饋
                    if image_count == 0:
                        st.warning("👀 這個 PDF 檔案裡面似乎沒有內嵌任何圖片檔喔！(如果是純掃描檔，可能需要另外的 OCR 技術)")
                    else:
                        st.success(f"🎉 成功！總共偵測並提取了 {image_count} 張圖片。")

                        # 5. 貼心的預覽功能 (利用 expander 折疊，保持畫面乾淨)
                        with st.expander("點擊預覽提取的圖片 (最多顯示前 3 張)"):
                            # 動態產生欄位
                            cols = st.columns(min(3, image_count))
                            for i, (name, img_data) in enumerate(extracted_images[:3]):
                                with cols[i]:
                                    st.image(img_data, caption=name, width='stretch')

                        # 6. 在記憶體中直接將圖片打包成 ZIP 檔
                        zip_buffer = io.BytesIO()
                        with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as zip_file:
                            for name, img_data in extracted_images:
                                zip_file.writestr(name, img_data)

                        # 將指針移回開頭供下載讀取
                        zip_buffer.seek(0)

                        st.write("---")
                        
                        # 7. 提供下載按鈕
                        st.download_button(
                            label="💾 下載所有圖片 (ZIP 壓縮檔)",
                            data=zip_buffer,
                            file_name=f"extracted_images_{uploaded_pdf.name}.zip",
                            mime="application/zip",
                            key="download_zip_btn"
                        )
                        
                except Exception as e:
                    # 捕捉解析損壞 PDF 等非預期錯誤
                    st.error(f"❌ 解析 PDF 時發生錯誤: {e}")
    st.sidebar.write("---")
    st.sidebar.write("權限狀態：一般使用者")

# --- 主程式入口邏輯 ---
if "logged_in" not in st.session_state or not st.session_state["logged_in"]:
    login_page()
else:  
    # 進入主功能頁面
    user_dashboard()