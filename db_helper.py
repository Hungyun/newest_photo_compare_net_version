import sqlite3
from datetime import datetime, timedelta

DB_FILE = "security.db"
MAX_ATTEMPTS = 3
LOCKOUT_DURATION = 15  # 鎖定時間（分鐘）

def init_db():
    """初始化資料庫，建立『鎖定追蹤』與『歷史日誌』兩張資料表"""
    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()
    
    # 資料表 1：用來追蹤當前鎖定狀態（資料會被複寫或歸零）
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS lockout_tracker (
            username TEXT PRIMARY KEY,
            failed_count INTEGER DEFAULT 0,
            last_failed_time TEXT
        )
    """)
    
    # 資料表 2：歷史流水帳日誌（只會新增，絕不複寫，用於資安審查）
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS login_logs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT,
            status TEXT,          -- 'SUCCESS' (成功), 'FAILED' (失敗), 'BLOCKED' (被鎖定阻擋)
            reason TEXT,          -- 紀錄詳細原因，例如 'LDAP 驗證失敗', '已達錯誤上限鎖定中'
            ip_address TEXT,      -- 登入者的 IP
            timestamp TEXT        -- 發生時間
        )
    """)
    
    conn.commit()
    conn.close()

def log_event(username, status, reason, ip_address):
    """公用工具：寫入一筆歷史流水帳到 login_logs 資料表"""
    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()
    current_time_str = datetime.now().isoformat()
    
    cursor.execute("""
        INSERT INTO login_logs (username, status, reason, ip_address, timestamp)
        VALUES (?, ?, ?, ?, ?)
    """, (username, status, reason, ip_address, current_time_str))
    
    conn.commit()
    conn.close()

def get_lockout_status(username):
    """檢查帳號鎖定狀態"""
    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()
    cursor.execute("SELECT failed_count, last_failed_time FROM lockout_tracker WHERE username = ?", (username,))
    row = cursor.fetchone()
    conn.close()

    if not row:
        return False, ""

    failed_count, last_failed_time_str = row
    
    if failed_count >= MAX_ATTEMPTS and last_failed_time_str:
        last_failed_time = datetime.fromisoformat(last_failed_time_str)
        lockout_deadline = last_failed_time + timedelta(minutes=LOCKOUT_DURATION)
        current_time = datetime.now()

        if current_time < lockout_deadline:
            time_diff = lockout_deadline - current_time
            minutes = int(time_diff.total_seconds() // 60)
            seconds = int(time_diff.total_seconds() % 60)
            return True, f"{minutes} 分 {seconds} 秒"
        else:
            # 鎖定時間已過，自動重設該帳號，但不另外寫日誌（等他下一次行為再記錄）
            reset_lockout(username)
            return False, ""
            
    return False, ""

def record_failed_attempt(username, ip_address):
    """記錄一次登入失敗，寫入歷史日誌，並傳回目前累計錯誤次數"""
    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()
    current_time_str = datetime.now().isoformat()

    # 1. 更新鎖定計數器
    cursor.execute("SELECT failed_count FROM lockout_tracker WHERE username = ?", (username,))
    row = cursor.fetchone()

    if row:
        new_count = row[0] + 1
        cursor.execute("""
            UPDATE lockout_tracker 
            SET failed_count = ?, last_failed_time = ? 
            WHERE username = ?
        """, (new_count, current_time_str, username))
    else:
        new_count = 1
        cursor.execute("""
            INSERT INTO lockout_tracker (username, failed_count, last_failed_time) 
            VALUES (?, ?, ?)
        """, (username, new_count, current_time_str))

    conn.commit()
    conn.close()

    # 2. 同步寫入歷史日誌 (判斷是否觸發鎖定)
    reason = "帳號密碼錯誤"
    if new_count >= MAX_ATTEMPTS:
        reason += f"（已達 {MAX_ATTEMPTS} 次上限，帳號鎖定 {LOCKOUT_DURATION} 分鐘）"
        
    log_event(username, "FAILED", reason, ip_address)
    
    return new_count

def reset_lockout(username):
    """單純清除鎖定追蹤計數（供內部呼叫或成功時重設）"""
    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()
    cursor.execute("""
        UPDATE lockout_tracker 
        SET failed_count = 0, last_failed_time = NULL 
        WHERE username = ?
    """, (username,))
    conn.commit()
    conn.close()