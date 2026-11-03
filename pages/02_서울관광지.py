
import streamlit as st
import folium
from streamlit_folium import st_folium

st.set_page_config(
    page_title="서울 인기 관광지 TOP 10",
    page_icon="🗺️",
    layout="wide"
)

# 서울 관광지 데이터
PLACES = [
    {
        "rank": 1,
        "name": "명동",
        "english": "Myeong-dong",
        "lat": 37.5636,
        "lon": 126.9850,
        "emoji": "🛍️",
        "category": "쇼핑 · 먹거리",
        "description": "K-뷰티 매장과 패션 상점, 길거리 음식이 모여 있는 쇼핑 명소예요."
    },
    {
        "rank": 2,
        "name": "N서울타워",
        "english": "N Seoul Tower",
        "lat": 37.5512,
        "lon": 126.9882,
        "emoji": "🌃",
        "category": "전망 · 야경",
        "description": "남산 정상에서 서울의 풍경을 한눈에 볼 수 있는 전망 명소예요."
    },
    {
        "rank": 3,
        "name": "코엑스·별마당도서관",
        "english": "COEX & Starfield Library",
        "lat": 37.5125,
        "lon": 127.0590,
        "emoji": "📚",
        "category": "쇼핑 · 문화",
        "description": "대형 쇼핑몰과 아름다운 별마당도서관이 있는 복합문화공간이에요."
    },
    {
        "rank": 4,
        "name": "강남역 일대",
        "english": "Gangnam",
        "lat": 37.4979,
        "lon": 127.0276,
        "emoji": "✨",
        "category": "도심 · 쇼핑",
        "description": "트렌디한 카페와 음식점, 쇼핑 공간이 많은 활기찬 거리예요."
    },
    {
        "rank": 5,
        "name": "홍대 거리",
        "english": "Hongdae",
        "lat": 37.5563,
        "lon": 126.9236,
        "emoji": "🎸",
        "category": "문화 · 공연",
        "description": "버스킹과 인디 음악, 개성 있는 상점으로 유명한 거리예요."
    },
    {
        "rank": 6,
        "name": "경복궁",
        "english": "Gyeongbokgung Palace",
        "lat": 37.5796,
        "lon": 126.9770,
        "emoji": "🏯",
        "category": "역사 · 전통",
        "description": "조선 왕조의 대표 궁궐로 아름다운 전통 건축을 감상할 수 있어요."
    },
    {
        "rank": 7,
        "name": "동대문디자인플라자",
        "english": "DDP",
        "lat": 37.5671,
        "lon": 127.0095,
        "emoji": "🎨",
        "category": "건축 · 전시",
        "description": "독특한 곡선형 건축물이 돋보이는 디자인과 전시 공간이에요."
    },
    {
        "rank": 8,
        "name": "광장시장",
        "english": "Gwangjang Market",
        "lat": 37.5700,
        "lon": 126.9997,
        "emoji": "🥟",
        "category": "전통시장 · 음식",
        "description": "빈대떡과 김밥 등 다양한 한국 전통 음식을 맛볼 수 있어요."
    },
    {
        "rank": 9,
        "name": "북촌한옥마을",
        "english": "Bukchon Hanok Village",
        "lat": 37.5826,
        "lon": 126.9830,
        "emoji": "🏘️",
        "category": "전통 · 산책",
        "description": "전통 한옥이 모여 있는 아름다운 동네로 옛 서울의 분위기를 느낄 수 있어요."
    },
    {
        "rank": 10,
        "name": "서울숲",
        "english": "Seoul Forest",
        "lat": 37.5444,
        "lon": 127.0374,
        "emoji": "🌳",
        "category": "자연 · 휴식",
        "description": "도심 속 넓은 공원으로 산책과 휴식을 즐기기 좋은 장소예요."
    }
]

# 제목
st.title("🗺️ 외국인이 좋아하는 서울 관광지 TOP 10")
st.write("서울의 인기 관광지를 지도에서 확인해 보세요!")

st.info(
    "📍 지도에 표시된 숫자를 클릭하면 "
    "관광지의 이름과 설명을 확인할 수 있어요!"
)

# 지도 생성
seoul_map = folium.Map(
    location=[37.5665, 126.9780],
    zoom_start=12,
    tiles="OpenStreetMap"
)

# 관광지 마커 표시
for place in PLACES:

    popup_html = f"""
    <div style="width:220px">
        <h4>
            {place['emoji']} {place['rank']}.
            {place['name']}
        </h4>
        <p>{place['description']}</p>
    </div>
    """

    folium.Marker(
        location=[
            place["lat"],
            place["lon"]
        ],
        tooltip=place["name"],
        popup=folium.Popup(
            popup_html,
            max_width=280
        ),
        icon=folium.DivIcon(
            html=f"""
            <div style="
                background:#2563eb;
                color:white;
                border:2px solid white;
                border-radius:50%;
                width:30px;
                height:30px;
                line-height:26px;
                text-align:center;
                font-weight:bold;
                font-size:14px;
            ">
                {place['rank']}
            </div>
            """,
            icon_size=(30, 30),
            icon_anchor=(15, 15)
        )
    ).add_to(seoul_map)

# 모든 관광지가 보이도록 지도 범위 설정
bounds = [
    [p["lat"], p["lon"]]
    for p in PLACES
]

seoul_map.fit_bounds(bounds, padding=(25, 25))

# 지도 화면 출력
st.subheader("📍 서울 관광지 위치")

st_folium(
    seoul_map,
    use_container_width=True,
    height=560,
    returned_objects=[]
)

# 관광지 설명
st.subheader("🎒 서울 관광지 TOP 10 소개")

st.caption("지도에 표시된 번호와 아래 관광지 번호가 같아요.")

# 2열로 관광지 소개
for start in range(0, len(PLACES), 2):

    cols = st.columns(2)

    for col, place in zip(
        cols,
        PLACES[start:start + 2]
    ):

        with col:

            with st.container(border=True):

                st.markdown(
                    f"### {place['emoji']} "
                    f"{place['rank']}. {place['name']}"
                )

                st.caption(
                    f"{place['english']} · "
                    f"{place['category']}"
                )

                st.write(place["description"])

                st.link_button(
                    "📍 구글 지도에서 위치 보기",
                    f"https://www.google.com/maps/search/?api=1&query={place['lat']},{place['lon']}",
                    use_container_width=True
                )

# 하단 안내
st.divider()

st.markdown(
    "🌏 **서울에서 즐거운 여행을 시작해 보세요!**"
)

st.caption(
    "지도 데이터: OpenStreetMap"
)

st.markdown(
    "[서울시 외국인 관광객 분석 자료 보기]"
    "(https://english.seoul.go.kr/from-k-culture-to-k-hiking-foreign-tourists-venture-beyond-seouls-top-attractions/)"
)
This app has encountered an error. The original error message is redacted to prevent data leaks. Full error details have been recorded in the logs (if you're on Streamlit Cloud, click on 'Manage app' in the lower right of your app).
Traceback:
File "/mount/src/aiproject/pages/03_관광지.py", line 203, in <module>
    st.divider()
    ^^
