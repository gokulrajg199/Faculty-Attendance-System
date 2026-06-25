import os
import re
import base64
import smtplib
from email.message import EmailMessage
from datetime import datetime, time

import cv2
import pandas as pd
import streamlit as st
from ultralytics import YOLO

st.set_page_config(page_title="SRIT Faculty Attendance", layout="wide")

# =========================
# FILE SETTINGS
# =========================
EXCEL_FILE = "Faculty name list 25-26.xlsx"
ATTENDANCE_FILE = "attendance.xlsx"
OD_FILE = "od_requests.xlsx"
PHOTO_DIR = "attendance_photos"
FACULTY_PHOTO_DIR = "faculty_photos"
OD_DOC_DIR = "od_documents"
LOGO_FILE = "college_logo.png"
HEADER_FILE = "header_banner.png"
YOLO_MODEL_FILE = "yolov8n.pt"

os.makedirs(PHOTO_DIR, exist_ok=True)
os.makedirs(FACULTY_PHOTO_DIR, exist_ok=True)
os.makedirs(OD_DOC_DIR, exist_ok=True)

# =========================
# EMAIL / ADMIN SETTINGS
# =========================
ALERT_EMAIL = "gokulraj.cse@sritcbe.ac.in"
SENDER_EMAIL = "gokulraj.cse@sritcbe.ac.in"
APP_PASSWORD = os.getenv("APP_PASSWORD", "").replace(" ", "")
ADMIN_PASSWORD = os.getenv("ADMIN_PASSWORD", "admin123")

# =========================
# TIME RULES
# =========================
MONTHLY_PERMISSION_LIMIT = 2
NORMAL_IN_START = time(8, 45)
NORMAL_IN_END = time(8, 47)
MORNING_PERMISSION_START = time(8, 48)
MORNING_PERMISSION_END = time(9, 47)
MORNING_HALF_DAY_START = time(9, 48)
MORNING_HALF_DAY_END = time(14, 2)
AFTERNOON_HALF_DAY_OUT_START = time(13, 0)
AFTERNOON_HALF_DAY_OUT_END = time(13, 2)
EVENING_PERMISSION_START = time(15, 50)
EVENING_PERMISSION_END = time(16, 49)
NORMAL_OUT_START = time(16, 50)
NORMAL_OUT_END = time(16, 52)

# =========================
# UI STYLE
# =========================
def b64_file(path: str) -> str:
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode()


def set_background():
    bg_css = ""
    if os.path.exists(LOGO_FILE):
        logo_base64 = b64_file(LOGO_FILE)
        bg_css = f'''
        background-image:
            linear-gradient(rgba(245,248,255,0.84), rgba(245,248,255,0.84)),
            url("data:image/png;base64,{logo_base64}");
        background-repeat:no-repeat;
        background-position:center;
        background-size:850px;
        background-attachment:fixed;
        '''
    st.markdown(
        f'''
        <style>
        .stApp {{{bg_css}}}
        .main-title {{text-align:center;color:#003366;font-size:34px;font-weight:900;margin-top:8px;margin-bottom:4px;}}
        .sub-title {{text-align:center;color:#1f4e79;font-size:16px;margin-bottom:18px;}}
        [data-testid="stSidebar"] {{background:linear-gradient(180deg,#002b5c,#004080);}}
        [data-testid="stSidebar"] * {{color:white !important;}}
        .stButton > button {{background:linear-gradient(90deg,#003366,#0077b6);color:white;border:none;border-radius:10px;padding:9px 16px;font-weight:700;box-shadow:0 4px 12px rgba(0,0,0,.18);}}
        .stButton > button:hover {{background:linear-gradient(90deg,#00509e,#00a6fb);color:white;}}
        [data-testid="stCameraInput"] {{max-width:420px;margin:auto;border-radius:18px;}}
        [data-testid="stDataFrame"] {{border-radius:14px;box-shadow:0 4px 18px rgba(0,0,0,.10);}}
        div[data-testid="stTabs"] button {{background:rgba(255,255,255,.86);border-radius:12px;padding:10px 18px;font-weight:800;color:#003366;}}
        div[data-testid="stTabs"] button:hover {{background:#dceeff;color:#002b5c;}}
        .camera-card {{background:rgba(255,255,255,.88);padding:22px;border-radius:20px;box-shadow:0 6px 24px rgba(0,0,0,.14);border:1px solid rgba(0,51,102,.15);margin-top:15px;margin-bottom:15px;text-align:center;}}
        </style>
        ''',
        unsafe_allow_html=True,
    )


set_background()

if os.path.exists(HEADER_FILE):
    header_base64 = b64_file(HEADER_FILE)
    st.markdown(
        f'''
        <div style="text-align:center;margin-bottom:10px;">
            <img src="data:image/png;base64,{header_base64}" style="width:95%;max-width:1400px;">
        </div>
        ''',
        unsafe_allow_html=True,
    )

st.markdown("<div class='main-title'>Faculty Attendance Management System</div>", unsafe_allow_html=True)
st.markdown("<div class='sub-title'>Powered by YOLO Computer Vision | SRIT Smart Campus Automation</div>", unsafe_allow_html=True)
st.write("---")

# =========================
# SESSION STATE
# =========================
if "camera_key" not in st.session_state:
    st.session_state.camera_key = 0
if "last_action" not in st.session_state:
    st.session_state.last_action = ""

# =========================
# LOADERS
# =========================
@st.cache_resource
@st.cache_resource
def load_model():
    return YOLO("yolov8n.pt")

@st.cache_data
def load_faculty_data():
    if not os.path.exists(EXCEL_FILE):
        st.error(f"Faculty Excel file not found: {EXCEL_FILE}")
        st.stop()
    xls = pd.ExcelFile(EXCEL_FILE)
    all_data = []
    for sheet in xls.sheet_names:
        raw = pd.read_excel(EXCEL_FILE, sheet_name=sheet, header=None)
        header_rows = raw[raw.apply(lambda row: row.astype(str).str.contains("EMP", case=False).any(), axis=1)].index
        if len(header_rows) == 0:
            continue
        df = pd.read_excel(EXCEL_FILE, sheet_name=sheet, header=header_rows[0])
        if {"EMP", "Name", "Designation"}.issubset(df.columns):
            df = df[["EMP", "Name", "Designation"]].dropna(subset=["EMP", "Name"])
            all_data.append(df)
    if not all_data:
        st.error("No valid faculty data found in Excel. Required columns: EMP, Name, Designation.")
        st.stop()
    final_df = pd.concat(all_data, ignore_index=True)
    return final_df.drop_duplicates(subset=["EMP", "Name"]).reset_index(drop=True)


model = load_model()
faculty_df = load_faculty_data()

# =========================
# DATA FUNCTIONS
# =========================
ATT_COLUMNS = [
    "Date", "EMP", "Name", "Designation", "In_Time", "Out_Time", "Status",
    "Permission_Type", "Monthly_Permission_Count", "In_Photo", "Out_Photo", "Remarks",
]
OD_COLUMNS = [
    "Request_ID", "Request_Date", "EMP", "Faculty_Name", "Designation",
    "OD_From_Date", "OD_To_Date", "OD_Purpose", "Approval_Status", "Approved_By",
    "Approval_Date", "Completion_Letter_Path", "Completion_Remarks", "Final_Status", "Attendance_Marked",
]


def read_attendance():
    if os.path.exists(ATTENDANCE_FILE):
        df = pd.read_excel(ATTENDANCE_FILE)
        for col in ATT_COLUMNS:
            if col not in df.columns:
                df[col] = ""
            df[col] = df[col].astype("object")
        return df[ATT_COLUMNS]
    return pd.DataFrame(columns=ATT_COLUMNS)


def save_attendance(df):
    df.to_excel(ATTENDANCE_FILE, index=False)


def read_od():
    if os.path.exists(OD_FILE):
        df = pd.read_excel(OD_FILE)
        for col in OD_COLUMNS:
            if col not in df.columns:
                df[col] = ""
            df[col] = df[col].astype("object")
        return df[OD_COLUMNS]
    return pd.DataFrame(columns=OD_COLUMNS)


def save_od(df):
    df.to_excel(OD_FILE, index=False)

# =========================
# EMAIL
# =========================
def send_mail(subject: str, body: str) -> bool:
    if not APP_PASSWORD:
        st.warning("Email not sent. APP_PASSWORD is not configured.")
        return False
    msg = EmailMessage()
    msg["Subject"] = subject
    msg["From"] = SENDER_EMAIL
    msg["To"] = ALERT_EMAIL
    msg.set_content(body)
    try:
        with smtplib.SMTP_SSL("smtp.gmail.com", 465) as smtp:
            smtp.login(SENDER_EMAIL, APP_PASSWORD)
            smtp.send_message(msg)
        st.success(f"Email notification sent to {ALERT_EMAIL}")
        return True
    except Exception as e:
        st.error(f"Mail sending failed: {e}")
        return False


def send_invalid_mail(emp, name, designation, entry_time, reason):
    body = f"""Attendance Alert

EMP ID: {emp}
Name: {name}
Designation: {designation}
Entry Time: {entry_time}
Reason: {reason}

Please verify this attendance entry.
"""
    return send_mail("SRIT Faculty Attendance Alert", body)


def send_od_mail(subject, row):
    body = f"""OD Request Update

Request ID: {row.get('Request_ID', '')}
EMP ID: {row.get('EMP', '')}
Name: {row.get('Faculty_Name', '')}
Designation: {row.get('Designation', '')}
OD From: {row.get('OD_From_Date', '')}
OD To: {row.get('OD_To_Date', '')}
Purpose: {row.get('OD_Purpose', '')}
Status: {row.get('Final_Status', '')}
"""
    return send_mail(subject, body)

# =========================
# BUSINESS LOGIC
# =========================
def permission_count(emp, month_year):
    df = read_attendance()
    if df.empty:
        return 0
    df["Date"] = df["Date"].astype(str)
    month_df = df[
        (df["EMP"].astype(str) == str(emp))
        & (df["Date"].str.endswith(month_year))
        & (df["Permission_Type"].astype(str).isin(["Morning Permission", "Evening Permission"]))
    ]
    return len(month_df)


def detect_person(frame):
    results = model.predict(source=frame, classes=[0], conf=0.40, imgsz=320, verbose=False)
    person_detected = False
    image_height, image_width = frame.shape[:2]
    for box in results[0].boxes:
        x1, y1, x2, y2 = box.xyxy[0]
        box_width = float(x2 - x1)
        box_height = float(y2 - y1)
        if (box_width / image_width) > 0.25 and (box_height / image_height) > 0.45:
            person_detected = True
    annotated = results[0].plot()
    return person_detected, annotated


def get_status_for_in(current_time, emp):
    month_year = datetime.now().strftime("%m-%Y")
    count = permission_count(emp, month_year)
    if NORMAL_IN_START <= current_time <= NORMAL_IN_END:
        return "Present", "None", count, "On-time present"
    if MORNING_PERMISSION_START <= current_time <= MORNING_PERMISSION_END:
        if count >= MONTHLY_PERMISSION_LIMIT:
            return "Half Day Leave", "Morning Half Day Leave", count, "Permission limit exceeded; converted to Morning Half Day Leave"
        return "Morning Permission", "Morning Permission", count + 1, "Morning permission used"
    if MORNING_HALF_DAY_START <= current_time <= MORNING_HALF_DAY_END:
        return "Half Day Leave", "Morning Half Day Leave", count, "Morning session leave; afternoon session present"
    return "Invalid Entry", "Invalid", count, "Invalid IN attendance time"


def update_status_for_out(row, current_time, emp):
    month_year = datetime.now().strftime("%m-%Y")
    count = permission_count(emp, month_year)
    if AFTERNOON_HALF_DAY_OUT_START <= current_time <= AFTERNOON_HALF_DAY_OUT_END:
        return "Half Day Leave", "Afternoon Half Day Leave", count, "Morning session present; afternoon session leave"
    if EVENING_PERMISSION_START <= current_time <= EVENING_PERMISSION_END:
        if count >= MONTHLY_PERMISSION_LIMIT:
            return "Half Day Leave", "Afternoon Half Day Leave", count, "Permission limit exceeded; converted to Afternoon Half Day Leave"
        return "Evening Permission", "Evening Permission", count + 1, "Evening permission used"
    if NORMAL_OUT_START <= current_time <= NORMAL_OUT_END:
        return "Present", "None", row.get("Monthly_Permission_Count", 0), "Full day present"
    if current_time > NORMAL_OUT_END:
        return "Present", "Late Exit", row.get("Monthly_Permission_Count", 0), "Late exit after college timing"
    return "Invalid Entry", "Invalid", count, "Invalid OUT attendance time"


def reset_camera():
    st.session_state.camera_key += 1
    st.rerun()

# =========================
# FACULTY PHOTO MATCHING
# =========================
def normalize_name(text: str) -> str:
    s = str(text).lower()
    for word in [
        "mr", "mrs", "ms", "dr", "prof", "ap", "cse", "it", "ece", "eee",
        "mech", "civil", "accounts", "account", "placement", "trainer", "hod",
        "principal", "and",
    ]:
        s = re.sub(rf"\b{word}\b", "", s)
    return re.sub(r"[^a-z0-9]", "", s)


def find_faculty_photo(selected_name: str):
    target = normalize_name(selected_name)
    if not os.path.exists(FACULTY_PHOTO_DIR):
        return None
    best_path = None
    best_score = 0
    for root, _, files in os.walk(FACULTY_PHOTO_DIR):
        for file in files:
            if not file.lower().endswith((".jpg", ".jpeg", ".png")):
                continue
            name = normalize_name(os.path.splitext(file)[0])
            score = 0
            if target == name:
                score = 100
            elif target in name or name in target:
                score = 90
            else:
                selected_tokens = set(re.findall(r"[a-z]+", str(selected_name).lower()))
                file_tokens = set(re.findall(r"[a-z]+", file.lower()))
                score = len(selected_tokens & file_tokens) * 20
            if score > best_score:
                best_score = score
                best_path = os.path.join(root, file)
    return best_path if best_score >= 20 else None


def show_uploaded_file(file_path):
    if file_path is None or str(file_path).strip() == "" or str(file_path).lower() in ["nan", "none"]:
        st.info("No document uploaded.")
        return
    file_path = str(file_path)
    if not os.path.exists(file_path):
        st.warning("Uploaded file path not found on this system.")
        return
    ext = os.path.splitext(file_path)[1].lower()
    if ext in [".jpg", ".jpeg", ".png"]:
        st.image(file_path, width=500)
    elif ext == ".pdf":
        with open(file_path, "rb") as pdf_file:
            st.download_button("View / Download PDF", data=pdf_file, file_name=os.path.basename(file_path), mime="application/pdf")
    else:
        with open(file_path, "rb") as doc_file:
            st.download_button("Download Uploaded File", data=doc_file, file_name=os.path.basename(file_path))

# =========================
# SIDEBAR
# =========================
portal = st.sidebar.radio("Select Portal", ["User Portal", "Admin Portal"])
is_admin = False
if portal == "Admin Portal":
    admin_password = st.sidebar.text_input("Admin Password", type="password")
    if admin_password == ADMIN_PASSWORD:
        is_admin = True
        st.sidebar.success("Admin Login Success")
    else:
        st.sidebar.warning("Enter Admin Password")

st.sidebar.header("Faculty Selection")
faculty_name = st.sidebar.selectbox("Select Faculty Name", faculty_df["Name"].astype(str).tolist())
selected = faculty_df[faculty_df["Name"].astype(str) == faculty_name].iloc[0]
st.sidebar.write("EMP:", selected["EMP"])
st.sidebar.write("Designation:", selected["Designation"])

# =========================
# DASHBOARD - ADMIN ONLY
# =========================
if portal == "Admin Portal" and is_admin:
    att_df_dash = read_attendance()
    today_text = datetime.now().strftime("%d-%m-%Y")
    total_faculty = len(faculty_df)
    today_df = att_df_dash[att_df_dash["Date"].astype(str) == today_text]
    present_df = today_df[today_df["Status"].astype(str) == "Present"]
    permission_df = today_df[today_df["Permission_Type"].astype(str).isin(["Morning Permission", "Evening Permission"])]
    halfday_df = today_df[today_df["Status"].astype(str) == "Half Day Leave"]
    od_today_df = today_df[today_df["Status"].astype(str) == "OD"]
    marked_emps = today_df["EMP"].astype(str).unique()
    absent_df = faculty_df[~faculty_df["EMP"].astype(str).isin(marked_emps)]
    st.markdown(f"""
        <div style="text-align:center;margin-top:10px;margin-bottom:20px;">
            <div style="font-size:28px;font-weight:900;color:#003366;">TODAY'S ATTENDANCE SUMMARY</div>
            <div style="font-size:17px;color:#1f4e79;">Date: {today_text} | Total Faculty: {total_faculty}</div>
        </div>
        """, unsafe_allow_html=True)
    c1, c2, c3, c4, c5 = st.columns(5)
    with c1:
        st.metric("🟢 Present", len(present_df))
    with c2:
        st.metric("🟡 Permission", len(permission_df))
    with c3:
        st.metric("🟠 Half Day", len(halfday_df))
    with c4:
        st.metric("🟣 OD", len(od_today_df))
    with c5:
        st.metric("🔴 Absent", len(absent_df))
    s1, s2, s3, s4, s5 = st.columns(5)
    with s1:
        show_present = st.button("View Present")
    with s2:
        show_permission = st.button("View Permission")
    with s3:
        show_halfday = st.button("View Half Day")
    with s4:
        show_od_today = st.button("View OD")
    with s5:
        show_absent = st.button("View Absent")
    if show_present:
        st.subheader("Today's Present Details")
        st.dataframe(present_df, use_container_width=True)
    if show_permission:
        st.subheader("Today's Permission Details")
        st.dataframe(permission_df, use_container_width=True)
    if show_halfday:
        st.subheader("Today's Half Day Details")
        st.dataframe(halfday_df, use_container_width=True)
    if show_od_today:
        st.subheader("Today's OD Details")
        st.dataframe(od_today_df, use_container_width=True)
    if show_absent:
        st.subheader("Today's Absent Details")
        st.dataframe(absent_df, use_container_width=True)
    st.write("---")

# =========================
# TABS
# =========================
if portal == "User Portal":
    tab1, user_od_tab = st.tabs(["Mark Attendance", "OD Request / Status"])
elif portal == "Admin Portal" and is_admin:
    tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs([
        "Mark Attendance",
        "Delete Wrong Attendance",
        "Search Attendance",
        "Daily Report",
        "Monthly Report",
        "Generate Absentees",
        "OD Management",
    ])
else:
    st.stop()

# =========================
# TAB 1 - MARK ATTENDANCE
# =========================
with tab1:
    st.markdown("<div class='camera-card'><h3 style='color:#003366;margin:0;'>📸 Webcam Attendance Capture</h3><p style='color:#1f4e79;margin:6px 0 0 0;'>Capture, verify with YOLO, and mark attendance securely</p></div>", unsafe_allow_html=True)
    if st.session_state.last_action:
        st.success(st.session_state.last_action)
    if st.button("Next Response / Next Person"):
        st.session_state.last_action = ""
        reset_camera()
    mode = st.radio("Select Attendance Type", ["Mark Morning / In Attendance", "Mark Exit / Out Attendance"])
    manual_time = st.time_input("Attendance Time", value=datetime.now().time().replace(second=0, microsecond=0))
    cam1, cam2, cam3 = st.columns([1, 2, 1])
    with cam2:
        photo = st.camera_input("Take faculty photo", key=f"camera_{st.session_state.camera_key}")
    if photo is not None:
        img_path = "captured_photo.jpg"
        with open(img_path, "wb") as f:
            f.write(photo.getbuffer())
        frame = cv2.imread(img_path)
        if frame is None:
            st.error("Captured image could not be read. Please retake the photo.")
            st.stop()
        person_detected, annotated = detect_person(frame)
        faculty_photo = find_faculty_photo(selected["Name"])
        img_col1, img_col2, img_col3 = st.columns(3)
        with img_col1:
            st.markdown("### 📸 Captured Photo")
            st.image(frame, channels="BGR", width=270)
        with img_col2:
            st.markdown("### 🤖 YOLO Detection")
            st.image(annotated, channels="BGR", width=270)
        with img_col3:
            st.markdown("### 👤 Registered Faculty Photo")
            if faculty_photo and os.path.exists(faculty_photo):
                st.image(faculty_photo, width=270)
            else:
                st.warning("Faculty photo not found")
        st.markdown("---")
        info1, info2, info3 = st.columns(3)
        with info1:
            st.metric("EMP ID", selected["EMP"])
        with info2:
            st.metric("Faculty Name", selected["Name"])
        with info3:
            st.metric("Designation", selected["Designation"])
        if not person_detected:
            st.error("No valid person detected. Attendance not marked.")
            st.stop()
        st.success("Clear person detected by YOLO")
        today = datetime.now().strftime("%d-%m-%Y")
        emp = selected["EMP"]
        att_df = read_attendance()
        today_row = att_df[(att_df["Date"].astype(str) == today) & (att_df["EMP"].astype(str) == str(emp))]
        if mode == "Mark Morning / In Attendance":
            if not today_row.empty:
                st.warning("IN attendance already marked for this faculty today. Click Next Response for next person.")
            elif st.button("Mark In Attendance"):
                current_time = manual_time
                current_time_text = current_time.strftime("%H:%M:%S")
                status, permission_type, permission_count_value, remarks = get_status_for_in(current_time, emp)
                if status in ["Invalid Entry", "Half Day Leave", "Morning Permission"]:
                    send_invalid_mail(emp, selected["Name"], selected["Designation"], current_time_text, remarks)
                if status == "Invalid Entry":
                    st.error("Invalid IN attendance time. Attendance not marked.")
                    st.stop()
                photo_filename = f"{today}_{current_time.strftime('%H-%M-%S')}_{emp}_IN.jpg"
                photo_path = os.path.join(PHOTO_DIR, photo_filename)
                cv2.imwrite(photo_path, frame)
                new_row = {
                    "Date": today,
                    "EMP": emp,
                    "Name": selected["Name"],
                    "Designation": selected["Designation"],
                    "In_Time": current_time_text,
                    "Out_Time": "",
                    "Status": status,
                    "Permission_Type": permission_type,
                    "Monthly_Permission_Count": permission_count_value,
                    "In_Photo": photo_path,
                    "Out_Photo": "",
                    "Remarks": remarks,
                }
                att_df = pd.concat([att_df, pd.DataFrame([new_row])], ignore_index=True)
                save_attendance(att_df)
                st.session_state.last_action = "IN Attendance Marked Successfully. Click Next Response for another person."
                reset_camera()
        else:
            if today_row.empty:
                st.warning("No IN attendance found. For morning leave, mark IN during the allowed afternoon half-day window.")
            else:
                existing_out = today_row.iloc[0]["Out_Time"]
                if existing_out not in ["", None] and not pd.isna(existing_out):
                    st.warning("OUT attendance already marked today. Click Next Response for next person.")
                elif st.button("Mark Out Attendance"):
                    current_time = manual_time
                    current_time_text = current_time.strftime("%H:%M:%S")
                    idx = today_row.index[0]
                    row = att_df.loc[idx].copy()
                    status, permission_type, permission_count_value, remarks = update_status_for_out(row, current_time, emp)
                    if status in ["Invalid Entry", "Half Day Leave", "Evening Permission", "Late Exit"]:
                        send_invalid_mail(emp, selected["Name"], selected["Designation"], current_time_text, remarks)
                    if status == "Invalid Entry":
                        st.error("Invalid OUT attendance time. Attendance not marked.")
                        st.stop()
                    photo_filename = f"{today}_{current_time.strftime('%H-%M-%S')}_{emp}_OUT.jpg"
                    photo_path = os.path.join(PHOTO_DIR, photo_filename)
                    cv2.imwrite(photo_path, frame)
                    att_df.loc[idx, "Out_Time"] = current_time_text
                    att_df.loc[idx, "Status"] = status
                    att_df.loc[idx, "Permission_Type"] = permission_type
                    att_df.loc[idx, "Monthly_Permission_Count"] = permission_count_value
                    att_df.loc[idx, "Out_Photo"] = photo_path
                    att_df.loc[idx, "Remarks"] = remarks
                    save_attendance(att_df)
                    st.session_state.last_action = "OUT Attendance Marked Successfully. Click Next Response for another person."
                    reset_camera()

# =========================
# USER PORTAL - OD REQUEST / STATUS
# =========================
if portal == "User Portal":
    with user_od_tab:
        st.subheader("OD Request / Status")
        od_df = read_od()
        st.markdown("### Submit New OD Request")
        od_from = st.date_input("OD From Date")
        od_to = st.date_input("OD To Date")
        od_purpose = st.text_area("OD Purpose")
        if st.button("Submit OD Request"):
            if od_to < od_from:
                st.error("OD To Date cannot be before OD From Date.")
            elif not od_purpose.strip():
                st.warning("Please enter OD purpose.")
            else:
                request_id = f"OD{datetime.now().strftime('%Y%m%d%H%M%S')}"
                new_row = {
                    "Request_ID": request_id,
                    "Request_Date": datetime.now().strftime("%d-%m-%Y"),
                    "EMP": selected["EMP"],
                    "Faculty_Name": selected["Name"],
                    "Designation": selected["Designation"],
                    "OD_From_Date": str(od_from),
                    "OD_To_Date": str(od_to),
                    "OD_Purpose": od_purpose,
                    "Approval_Status": "Pending Approval",
                    "Approved_By": "",
                    "Approval_Date": "",
                    "Completion_Letter_Path": "",
                    "Completion_Remarks": "",
                    "Final_Status": "Pending Approval",
                    "Attendance_Marked": "No",
                }
                od_df = pd.concat([od_df, pd.DataFrame([new_row])], ignore_index=True)
                save_od(od_df)
                send_od_mail("SRIT OD Request Submitted", new_row)
                st.success("OD Request Submitted Successfully")
        st.markdown("---")
        st.markdown("### My OD Status")
        my_od = od_df[od_df["EMP"].astype(str) == str(selected["EMP"])]
        st.dataframe(my_od, use_container_width=True)
        approved_df = my_od[my_od["Final_Status"].astype(str) == "Approved - Pending Completion"]
        if not approved_df.empty:
            st.markdown("### Upload OD Certificate / Completion Proof")
            req_id = st.selectbox("Select Approved OD Request", approved_df["Request_ID"].astype(str).tolist())
            cert_file = st.file_uploader("Upload OD Certificate / Completion Letter", type=["pdf", "jpg", "jpeg", "png", "doc", "docx"])
            completion_remarks = st.text_area("Completion Remarks")
            if st.button("Submit OD Certificate"):
                if cert_file is None:
                    st.warning("Please upload certificate.")
                else:
                    safe_name = re.sub(r"[^a-zA-Z0-9_.-]", "_", cert_file.name)
                    file_path = os.path.join(OD_DOC_DIR, f"{req_id}_{safe_name}")
                    with open(file_path, "wb") as f:
                        f.write(cert_file.getbuffer())
                    idx = od_df[od_df["Request_ID"].astype(str) == str(req_id)].index[0]
                    od_df.loc[idx, "Completion_Letter_Path"] = file_path
                    od_df.loc[idx, "Completion_Remarks"] = completion_remarks
                    od_df.loc[idx, "Final_Status"] = "Completion Submitted"
                    save_od(od_df)
                    send_od_mail("SRIT OD Certificate Submitted for Verification", od_df.loc[idx])
                    st.success("OD Certificate Submitted for Admin Verification")
                    st.rerun()

# =========================
# ADMIN PORTAL TABS
# =========================
if portal == "Admin Portal" and is_admin:
    with tab2:
        st.subheader("Delete Wrong Attendance")
        att_df = read_attendance()
        if len(att_df) > 0:
            st.dataframe(att_df, use_container_width=True)
            delete_index = st.number_input("Enter row number to delete", min_value=0, max_value=max(len(att_df) - 1, 0), step=1)
            if st.button("Delete Selected Attendance"):
                att_df = att_df.drop(index=delete_index).reset_index(drop=True)
                save_attendance(att_df)
                st.success("Selected attendance deleted successfully")
                st.rerun()
            if st.button("Delete Last Attendance"):
                att_df = att_df.iloc[:-1]
                save_attendance(att_df)
                st.success("Last attendance deleted successfully")
                st.rerun()
        else:
            st.info("No attendance records found.")

    with tab3:
        st.subheader("Search Attendance")
        att_df = read_attendance()
        search_name = st.text_input("Enter faculty name / EMP to search")
        if search_name:
            result = att_df[
                att_df["Name"].astype(str).str.contains(search_name, case=False, na=False)
                | att_df["EMP"].astype(str).str.contains(search_name, case=False, na=False)
            ]
            st.dataframe(result, use_container_width=True)
        else:
            st.dataframe(att_df, use_container_width=True)

    with tab4:
        st.subheader("Daily Attendance Report")
        att_df = read_attendance()
        selected_date = st.text_input("Enter date in DD-MM-YYYY format", datetime.now().strftime("%d-%m-%Y"))
        daily_df = att_df[att_df["Date"].astype(str) == selected_date]
        st.write("Total Records:", len(daily_df))
        st.dataframe(daily_df, use_container_width=True)
        daily_file = f"Daily_Report_{selected_date}.xlsx"
        daily_df.to_excel(daily_file, index=False)
        with open(daily_file, "rb") as f:
            st.download_button("Download Daily Report", data=f, file_name=daily_file, mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")

    with tab5:
        st.subheader("Monthly Attendance Report")
        att_df = read_attendance()
        month = st.text_input("Enter month MM-YYYY", datetime.now().strftime("%m-%Y"))
        monthly_df = att_df[att_df["Date"].astype(str).str.endswith(month)]
        st.write("Total Monthly Records:", len(monthly_df))
        st.dataframe(monthly_df, use_container_width=True)
        monthly_file = f"Monthly_Report_{month}.xlsx"
        monthly_df.to_excel(monthly_file, index=False)
        with open(monthly_file, "rb") as f:
            st.download_button("Download Monthly Report", data=f, file_name=monthly_file, mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")

    with tab6:
        st.subheader("Generate Absentees")
        selected_date = st.text_input("Absentee Date DD-MM-YYYY", datetime.now().strftime("%d-%m-%Y"))
        if st.button("Generate Absentees for Selected Date"):
            att_df = read_attendance()
            already_absent = att_df[(att_df["Date"].astype(str) == selected_date) & (att_df["Status"].astype(str) == "Absent")]
            if not already_absent.empty:
                st.warning("Absentees already generated for this date.")
            else:
                present_emps = att_df[att_df["Date"].astype(str) == selected_date]["EMP"].astype(str).tolist()
                absent_staff = faculty_df[~faculty_df["EMP"].astype(str).isin(present_emps)].copy()
                absent_records = []
                for _, row in absent_staff.iterrows():
                    absent_records.append({
                        "Date": selected_date,
                        "EMP": row["EMP"],
                        "Name": row["Name"],
                        "Designation": row["Designation"],
                        "In_Time": "",
                        "Out_Time": "",
                        "Status": "Absent",
                        "Permission_Type": "None",
                        "Monthly_Permission_Count": 0,
                        "In_Photo": "",
                        "Out_Photo": "",
                        "Remarks": "No attendance entry",
                    })
                if absent_records:
                    att_df = pd.concat([att_df, pd.DataFrame(absent_records)], ignore_index=True)
                    save_attendance(att_df)
                    st.success("Absentees generated successfully")
                    st.dataframe(pd.DataFrame(absent_records), use_container_width=True)
                else:
                    st.success("No absentees found.")

    with tab7:
        st.subheader("OD Management")
        od_menu = st.radio("Select OD Option", ["Approve / Reject OD", "Verify Certificate & Mark OD", "View OD Records"], horizontal=True)
        od_df = read_od()
        if od_menu == "Approve / Reject OD":
            st.markdown("### Approve / Reject OD Request")
            pending_df = od_df[od_df["Approval_Status"].astype(str) == "Pending Approval"]
            if pending_df.empty:
                st.info("No pending OD requests.")
            else:
                for i, row in pending_df.iterrows():
                    with st.expander(f"{row['Request_ID']} - {row['Faculty_Name']}"):
                        st.write("EMP:", row["EMP"])
                        st.write("Designation:", row["Designation"])
                        st.write("OD From:", row["OD_From_Date"])
                        st.write("OD To:", row["OD_To_Date"])
                        st.write("Purpose:", row["OD_Purpose"])
                        approve = st.checkbox("Approve this OD", key=f"approve_{row['Request_ID']}")
                        reject = st.checkbox("Reject this OD", key=f"reject_{row['Request_ID']}")
                        approved_by = st.text_input("Approved / Rejected By", "HOD/Admin", key=f"by_{row['Request_ID']}")
                        remarks = st.text_area("Approval Remarks", key=f"remarks_{row['Request_ID']}")
                        if st.button("Submit Decision", key=f"submit_{row['Request_ID']}"):
                            if approve and reject:
                                st.error("Select only one option: Approve or Reject.")
                            elif approve:
                                od_df.loc[i, "Approval_Status"] = "Approved - Pending Completion"
                                od_df.loc[i, "Final_Status"] = "Approved - Pending Completion"
                                od_df.loc[i, "Approved_By"] = approved_by
                                od_df.loc[i, "Approval_Date"] = datetime.now().strftime("%d-%m-%Y")
                                od_df.loc[i, "Completion_Remarks"] = remarks
                                save_od(od_df)
                                send_od_mail("SRIT OD Request Approved", od_df.loc[i])
                                st.success("OD Approved Successfully")
                                st.rerun()
                            elif reject:
                                od_df.loc[i, "Approval_Status"] = "Rejected"
                                od_df.loc[i, "Final_Status"] = "Rejected"
                                od_df.loc[i, "Approved_By"] = approved_by
                                od_df.loc[i, "Approval_Date"] = datetime.now().strftime("%d-%m-%Y")
                                od_df.loc[i, "Completion_Remarks"] = remarks
                                save_od(od_df)
                                send_od_mail("SRIT OD Request Rejected", od_df.loc[i])
                                st.warning("OD Rejected")
                                st.rerun()
                            else:
                                st.warning("Please select Approve or Reject.")
        elif od_menu == "Verify Certificate & Mark OD":
            st.markdown("### Verify Uploaded Certificate and Mark Attendance as OD")
            completion_df = od_df[od_df["Final_Status"].astype(str) == "Completion Submitted"]
            if completion_df.empty:
                st.info("No OD certificate pending verification.")
            else:
                request_id = st.selectbox("Select OD Request", completion_df["Request_ID"].astype(str).tolist())
                selected_od = completion_df[completion_df["Request_ID"].astype(str) == str(request_id)].iloc[0]
                st.write("Faculty:", selected_od["Faculty_Name"])
                st.write("EMP:", selected_od["EMP"])
                st.write("Designation:", selected_od["Designation"])
                st.write("OD From:", selected_od["OD_From_Date"])
                st.write("OD To:", selected_od["OD_To_Date"])
                st.write("Purpose:", selected_od["OD_Purpose"])
                st.markdown("### Uploaded Certificate")
                show_uploaded_file(selected_od["Completion_Letter_Path"])
                if st.button("Verify Certificate and Mark OD Attendance"):
                    idx = od_df[od_df["Request_ID"].astype(str) == str(request_id)].index[0]
                    att_df = read_attendance()
                    od_dates = pd.date_range(start=pd.to_datetime(selected_od["OD_From_Date"]), end=pd.to_datetime(selected_od["OD_To_Date"]))
                    for od_date in od_dates:
                        date_text = od_date.strftime("%d-%m-%Y")
                        existing = att_df[(att_df["Date"].astype(str) == date_text) & (att_df["EMP"].astype(str) == str(selected_od["EMP"]))]
                        if existing.empty:
                            new_att = {
                                "Date": date_text,
                                "EMP": selected_od["EMP"],
                                "Name": selected_od["Faculty_Name"],
                                "Designation": selected_od["Designation"],
                                "In_Time": "",
                                "Out_Time": "",
                                "Status": "OD",
                                "Permission_Type": "Official Duty",
                                "Monthly_Permission_Count": 0,
                                "In_Photo": "",
                                "Out_Photo": "",
                                "Remarks": "OD certificate verified and attendance marked as OD",
                            }
                            att_df = pd.concat([att_df, pd.DataFrame([new_att])], ignore_index=True)
                    save_attendance(att_df)
                    od_df.loc[idx, "Final_Status"] = "OD Completed"
                    od_df.loc[idx, "Attendance_Marked"] = "Yes"
                    save_od(od_df)
                    send_od_mail("SRIT OD Completed and Attendance Marked", od_df.loc[idx])
                    st.success("Certificate Verified. Attendance Marked as OD.")
                    st.rerun()
        elif od_menu == "View OD Records":
            st.markdown("### OD Records")
            st.dataframe(od_df, use_container_width=True)
            if os.path.exists(OD_FILE):
                with open(OD_FILE, "rb") as f:
                    st.download_button("Download OD Records", data=f, file_name="od_requests.xlsx", mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")

# =========================
# FINAL ATTENDANCE RECORDS - ADMIN ONLY
# =========================
if portal == "Admin Portal" and is_admin:
    st.write("---")
    st.subheader("Complete Attendance Records")
    final_att_df = read_attendance()
    st.dataframe(final_att_df, use_container_width=True)
    if os.path.exists(ATTENDANCE_FILE):
        with open(ATTENDANCE_FILE, "rb") as f:
            st.download_button("Download Full Attendance Excel", data=f, file_name="attendance.xlsx", mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
