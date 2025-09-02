# # import pandas as pd
# # import plotly.express as px
# # import streamlit as st
# # import xlwings as xw
# # from PIL import Image, ImageSequence
# # import glob
# # import geocoder
# # import time
# # import os
# # import datetime

# # st.set_page_config(layout="wide")

# # # Get last 6 distress images
# # distress_images = sorted(glob.glob(
# #     r'C:\distress_signal_using_cv-master\distress_signal_using_cv-master\distress_signal_using_cv1-main\data\frame*.jpg'))[-6:]

# # # Get last 4 weapon images
# # weapon_images = sorted(glob.glob(
# #     r'C:\distress_signal_using_cv-master\distress_signal_using_cv-master\distress_signal_using_cv1-main\wep_img\wep_*.jpg'))[-4:]

# # # Get GIF frames
# # gif_frame_paths = sorted(
# #     [f for f in glob.glob(
# #         r'C:\distress_signal_using_cv-master\distress_signal_using_cv-master\distress_signal_using_cv1-main\data\frame*.jpg')
# #      if int(os.path.basename(f).replace("frame", "").replace(".jpg", "")) % 20 == 0]
# # )

# # # Create GIF
# # gif_path = "data\output.gif"
# # if gif_frame_paths:
# #     frames = [Image.open(f) for f in gif_frame_paths]
# #     frames[0].save(gif_path, format="GIF", save_all=True, append_images=frames[1:], duration=200, loop=0)

# # header_left, header_mid, header_right = st.columns([3, 1, 1], gap='small')

# # with header_left:
# #     st.title(':yellow[Police Dashboard]')

# # filepath = r"data.xlsx"

# # col1, col2 = st.columns(2)

# # # Open Excel file without showing the application
# # app = xw.App(visible=False)
# # ws = app.books.open(filepath).sheets['sheet']

# # x = ws.range("A1").value
# # y = ws.range("B1").value

# # # Real-time location tracking
# # g = geocoder.ip('me')  # Get the current location using IP geolocation
# # lat, lng = g.latlng  # Get latitude and longitude

# # # Create a DataFrame with correct column names for st.map
# # location_data = pd.DataFrame({
# #     'LAT': [lat],
# #     'LON': [lng]
# # })

# # with col1:
# #     st.title(':red[current status:]')

# #     if x + y == 2:
# #         st.header(':red[Potential Threat Detected!]')
# #         st.subheader("Attention Needed")
# #         st.write("Distress Signal Detected in your Vicinity\n\n\n")
# #         st.divider()

# #         # Display 6 distress images when a threat is detected
# #         if distress_images:
# #             st.header(':red[Snapshots]')
# #             cols = st.columns(2)
# #             for i, img_path in enumerate(distress_images):
# #                 with cols[i % 2]:
# #                     image = Image.open(img_path)
# #                     # Get file save time
# #                     timestamp = os.path.getmtime(img_path)
# #                     time_str = datetime.datetime.fromtimestamp(timestamp).strftime('%Y-%m-%d %H:%M:%S')
# #                     st.image(image, width=200)
# #                     st.caption(f"Captured: {time_str}")

# #         # Display 4 weapon images if available
# #         if weapon_images:
# #             st.header(':red[Weapon]')
# #             cols = st.columns(2)
# #             for i, img_path in enumerate(weapon_images):
# #                 with cols[i % 2]:
# #                     image = Image.open(img_path)
# #                     # Get file save time
# #                     timestamp = os.path.getmtime(img_path)
# #                     time_str = datetime.datetime.fromtimestamp(timestamp).strftime('%Y-%m-%d %H:%M:%S')
# #                     st.image(image, width=200)
# #                     st.caption(f"Captured: {time_str}")

# #         # Display the generated GIF
# #         if os.path.exists(gif_path):
# #             st.header(":red[Playback]")
# #             st.image(gif_path)

# #     elif x + y == 1:
# #         st.header("New Alert Detected!!")
# #         st.divider()

# #     else:
# #         st.header(':yellow[No Attention Required]')

# # # Display map with the real-time location
# # st.header(':red[Location]')
# # st.map(location_data)

# # # Close workbook and quit Excel application
# # ws.book.close()
# # app.quit()






# # # import pandas as pd
# # # import plotly.express as px
# # # import streamlit as st
# # # import xlwings as xw
# # # from PIL import Image
# # # import glob
# # # import geocoder
# # # import time

# # # st.set_page_config(layout="wide")

# # # # Get last 6 distress images
# # # distress_images = sorted(glob.glob(
# # #     r'C:\distress_signal_using_cv-master\distress_signal_using_cv-master\distress_signal_using_cv1-main\data\frame*.jpg'))[-6:]

# # # # Get last 4 weapon images
# # # weapon_images = sorted(glob.glob(
# # #     r'C:\distress_signal_using_cv-master\distress_signal_using_cv-master\distress_signal_using_cv1-main\wep_img\wep_*.jpg'))[-4:]

# # # header_left, header_mid, header_right = st.columns([3, 1, 1], gap='small')

# # # with header_left:
# # #     st.title(':yellow[Police Dashboard]')

# # # filepath = r"C:\distress_signal_using_cv-master\distress_signal_using_cv-master\distress_signal_using_cv1-main\data.xlsx"

# # # col1, col2 = st.columns(2)

# # # # Open Excel file without showing the application
# # # app = xw.App(visible=False)
# # # ws = app.books.open(filepath).sheets['sheet']

# # # x = ws.range("A1").value
# # # y = ws.range("B1").value

# # # # Real-time location tracking
# # # g = geocoder.ip('me')  # Get the current location using IP geolocation
# # # lat, lng = g.latlng  # Get latitude and longitude

# # # # Create a DataFrame with correct column names for st.map
# # # location_data = pd.DataFrame({
# # #     'LAT': [lat],
# # #     'LON': [lng]
# # # })

# # # with col1:
# # #     st.title(':red[current status:]')

# # #     if x + y == 2:
# # #         st.header(':red[Potential Threat Detected!]')
# # #         st.subheader("Attention Needed")
# # #         st.write("Distress Signal Detected in your Vicinity\n\n\n")
# # #         st.divider()

# # #         # Display 6 distress images when a threat is detected
# # #         if distress_images:
# # #             st.header(':red[Snapshots]')
# # #             cols = st.columns(2)  # 2 columns to display the distress images horizontally
# # #             for i, img_path in enumerate(distress_images):
# # #                 with cols[i % 2]:  # Alternate between the two columns
# # #                     image = Image.open(img_path)
# # #                     st.image(image, width=200)

# # #         # Display 4 weapon images if available
# # #         if weapon_images:
# # #             st.header(':red[Weapon]')
# # #             cols = st.columns(2)  # 2 columns for weapon images horizontally
# # #             for i, img_path in enumerate(weapon_images):
# # #                 with cols[i % 2]:
# # #                     image = Image.open(img_path)
# # #                     st.image(image, width=200)

# # #     elif x + y == 1:
# # #         st.header("New Alert Detected!!")
# # #         st.divider()

# # #     else:
# # #         st.header(':yellow[No Attention Required]')

# # # # Display map with the real-time location
# # # st.map(location_data)

# # # # Close workbook and quit Excel application
# # # ws.book.close()
# # # app.quit()

# #web.py

# import pandas as pd
# import plotly.express as px
# import streamlit as st
# import xlwings as xw
# from PIL import Image, ImageSequence
# import glob
# import geocoder
# import time
# import os
# import datetime

# st.set_page_config(layout="wide")

# # Get last 6 distress images
# distress_images = sorted(glob.glob(
#     r'data\frame*.jpg'))[-6:]

# # Get last 4 weapon images
# weapon_images = sorted(glob.glob(
#     r'wep_img\wep_*.jpg'))[-4:]

# # Get GIF frames
# gif_frame_paths = sorted(
#     [f for f in glob.glob(
#         r'data\frame*.jpg')
#      if int(os.path.basename(f).replace("frame", "").replace(".jpg", "")) % 20 == 0]
# )

# # Create GIF
# gif_path = "data\output.gif"
# if gif_frame_paths:
#     frames = [Image.open(f) for f in gif_frame_paths]
#     frames[0].save(gif_path, format="GIF", save_all=True, append_images=frames[1:], duration=200, loop=0)

# header_left, header_mid, header_right = st.columns([3, 1, 1], gap='small')

# with header_left:
#     st.title(':yellow[Police Dashboard]')

# filepath = r"data.xlsx"

# col1, col2 = st.columns(2)

# # Open Excel file without showing the application
# app = xw.App(visible=False)
# ws = app.books.open(filepath).sheets['sheet']

# x = ws.range("A1").value
# y = ws.range("B1").value

# # Real-time location tracking
# g = geocoder.ip('me')  # Get the current location using IP geolocation
# lat, lng = g.latlng  # Get latitude and longitude

# # Create a DataFrame with correct column names for st.map
# location_data = pd.DataFrame({
#     'LAT': [lat],
#     'LON': [lng]
# })

# with col1:
#     st.title(':red[current status:]')

#     if x + y == 2:
#         st.header(':red[Potential Threat Detected!]')
#         st.subheader("Attention Needed")
#         st.write("Distress Signal Detected in your Vicinity\n\n\n")
#         st.divider()

#         # Display 6 distress images when a threat is detected
#         if distress_images:
#             st.header(':red[Snapshots]')
#             cols = st.columns(2)
#             for i, img_path in enumerate(distress_images):
#                 with cols[i % 2]:
#                     image = Image.open(img_path)
#                     # Get file save time
#                     timestamp = os.path.getmtime(img_path)
#                     time_str = datetime.datetime.fromtimestamp(timestamp).strftime('%Y-%m-%d %H:%M:%S')
#                     st.image(image, width=200)
#                     st.caption(f"Captured: {time_str}")

#         # Display 4 weapon images if available
#         if weapon_images:
#             st.header(':red[Weapon]')
#             cols = st.columns(2)
#             for i, img_path in enumerate(weapon_images):
#                 with cols[i % 2]:
#                     image = Image.open(img_path)
#                     # Get file save time
#                     timestamp = os.path.getmtime(img_path)
#                     time_str = datetime.datetime.fromtimestamp(timestamp).strftime('%Y-%m-%d %H:%M:%S')
#                     st.image(image, width=200)
#                     st.caption(f"Captured: {time_str}")

#         # Display the generated GIF
#         if os.path.exists(gif_path):
#             st.header(":red[Playback]")
#             st.image(gif_path)

#     elif x + y == 1:
#         st.header("New Alert Detected!!")
#         st.divider()

#     else:
#         st.header(':yellow[No Attention Required]')

# # Display map with the real-time location
# st.header(':red[Location]')
# st.map(location_data)

# # Close workbook and quit Excel application
# ws.book.close()
# app.quit()






# # import pandas as pd
# # import plotly.express as px
# # import streamlit as st
# # import xlwings as xw
# # from PIL import Image 
# # import glob
# # import geocoder
# # import time

# # st.set_page_config(layout="wide")

# # # Get last 6 distress images
# # distress_images = sorted(glob.glob(
# #     r'C:\distress_signal_using_cv-master\distress_signal_using_cv-master\distress_signal_using_cv1-main\data\frame*.jpg'))[-6:]

# # # Get last 4 weapon images
# # weapon_images = sorted(glob.glob(
# #     r'C:\distress_signal_using_cv-master\distress_signal_using_cv-master\distress_signal_using_cv1-main\wep_img\wep_*.jpg'))[-4:]

# # header_left, header_mid, header_right = st.columns([3, 1, 1], gap='small')

# # with header_left:
# #     st.title(':yellow[Police Dashboard]')

# # filepath = r"C:\distress_signal_using_cv-master\distress_signal_using_cv-master\distress_signal_using_cv1-main\data.xlsx"

# # col1, col2 = st.columns(2)

# # # Open Excel file without showing the application
# # app = xw.App(visible=False)
# # ws = app.books.open(filepath).sheets['sheet']

# # x = ws.range("A1").value
# # y = ws.range("B1").value

# # # Real-time location tracking
# # g = geocoder.ip('me')  # Get the current location using IP geolocation
# # lat, lng = g.latlng  # Get latitude and longitude

# # # Create a DataFrame with correct column names for st.map
# # location_data = pd.DataFrame({
# #     'LAT': [lat],
# #     'LON': [lng]
# # })

# # with col1:
# #     st.title(':red[current status:]')

# #     if x + y == 2:
# #         st.header(':red[Potential Threat Detected!]')
# #         st.subheader("Attention Needed")
# #         st.write("Distress Signal Detected in your Vicinity\n\n\n")
# #         st.divider()

# #         # Display 6 distress images when a threat is detected
# #         if distress_images:
# #             st.header(':red[Snapshots]')
# #             cols = st.columns(2)  # 2 columns to display the distress images horizontally
# #             for i, img_path in enumerate(distress_images):
# #                 with cols[i % 2]:  # Alternate between the two columns
# #                     image = Image.open(img_path)
# #                     st.image(image, width=200)

# #         # Display 4 weapon images if available
# #         if weapon_images:
# #             st.header(':red[Weapon]')
# #             cols = st.columns(2)  # 2 columns for weapon images horizontally
# #             for i, img_path in enumerate(weapon_images):
# #                 with cols[i % 2]:
# #                     image = Image.open(img_path)
# #                     st.image(image, width=200)

# #     elif x + y == 1:
# #         st.header("New Alert Detected!!")
# #         st.divider()

# #     else:
# #         st.header(':yellow[No Attention Required]')

# # # Display map with the real-time location
# # st.map(location_data)

# # # Close workbook and quit Excel application
# # ws.book.close()
# # app.quit()

import pandas as pd
import streamlit as st
from PIL import Image
import glob
import os
import datetime
import geocoder
import xlwings as xw

# Set Streamlit page config
st.set_page_config(layout="wide")

# ------------------------ Helper Functions ------------------------ #
def get_images(path_pattern, limit):
    """Get the latest N images from the given glob pattern."""
    return sorted(glob.glob(path_pattern))[-limit:]

def create_gif(frames, output_path):
    """Create a GIF from a list of image paths."""
    if frames:
        images = [Image.open(f) for f in frames]
        images[0].save(output_path, format="GIF", save_all=True, append_images=images[1:], duration=200, loop=0)

def display_images(title, image_paths, cols_count=2):
    """Display images in Streamlit with captions."""
    st.header(title)
    cols = st.columns(cols_count)
    for i, path in enumerate(image_paths):
        with cols[i % cols_count]:
            img = Image.open(path)
            timestamp = os.path.getmtime(path)
            time_str = datetime.datetime.fromtimestamp(timestamp).strftime('%Y-%m-%d %H:%M:%S')
            st.image(img, width=200)
            st.caption(f"Captured: {time_str}")

# ------------------------ Load Resources ------------------------ #
distress_images = get_images('data/frame*.jpg', 6)
weapon_images = get_images('wep_img/wep_*.jpg', 4)

# Create GIF from every 20th frame
gif_frame_paths = [f for f in glob.glob('data/frame*.jpg') if int(os.path.basename(f).replace("frame", "").replace(".jpg", "")) % 20 == 0]
gif_path = "data/output.gif"
create_gif(gif_frame_paths, gif_path)

# ------------------------ UI Layout ------------------------ #
header_left, _, _ = st.columns([3, 1, 1], gap='small')
with header_left:
    st.title(':yellow[Police Dashboard]')

# ------------------------ Read Status from Excel ------------------------ #
filepath = "data.xlsx"
app = xw.App(visible=False)
ws = app.books.open(filepath).sheets['sheet']
x = ws.range("A1").value
y = ws.range("B1").value
ws.book.close()
app.quit()

# ------------------------ Get Real-time Location ------------------------ #
try:
    g = geocoder.ip('me')
    lat, lng = g.latlng if g.latlng else (0.0, 0.0)
except:
    lat, lng = (0.0, 0.0)

location_data = pd.DataFrame({'LAT': [lat], 'LON': [lng]})

# ------------------------ Current Status ------------------------ #
col1, _ = st.columns(2)
with col1:
    st.title(':red[current status:]')

    if x + y == 2:
        st.header(':red[Potential Threat Detected!]')
        st.subheader("Attention Needed")
        st.write("Distress Signal Detected in your Vicinity")
        st.divider()

        if distress_images:
            display_images(':red[Snapshots]', distress_images)

        if weapon_images:
            display_images(':red[Weapon]', weapon_images)

        if os.path.exists(gif_path):
            st.header(":red[Playback]")
            st.image(gif_path)

    elif x + y == 1:
        st.header("New Alert Detected!!")
        st.divider()
    else:
        st.header(':yellow[No Attention Required]')

# ------------------------ Show Map ------------------------ #
st.header(':red[Location]')
st.map(location_data)
