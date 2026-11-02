
import streamlit as st

# =====================================
# 1. 페이지 설정
# =====================================

st.set_page_config(
    page_title="MBTI 진로 탐험소",
    page_icon="🌈",
    layout="wide"
)

# =====================================
# 2. 화면 디자인
# =====================================

st.markdown("""
<style>
.stApp {
    background-color: #F5F7FF;
}

.main-title {
    text-align: center;
    font-size: 38px;
    font-weight: bold;
    color: #6554C0;
    padding: 20px;
}

.sub-title {
    text-align: center;
    color: #777777;
    font-size: 18px;
    margin-bottom: 30px;
}

.job-card {
    background-color: white;
    padding: 25px;
    border-radius: 18px;
    border: 2px solid #E6E0FF;
    margin-bottom: 15px;
    min-height: 310px;
}

.job-name {
    color: #6554C0;
    font-size: 25px;
    font-weight: bold;
}

.info {
    color: #444444;
    font-size: 16px;
    line-height: 1.8;
}

.footer {
    text-align: center;
    color: gray;
    padding: 20px;
}
</style>
""", unsafe_allow_html=True)

# =====================================
# 3. MBTI별 진로 데이터
# =====================================

career_data = {
    "ISTJ": [
        {
            "job": "회계사",
            "major": "회계학과, 경영학과, 경제학과",
            "personality": "꼼꼼하고 책임감이 강하며 숫자와 자료를 정확하게 다루는 사람",
            "salary": "약 5,000만~8,000만 원"
        },
        {
            "job": "공무원",
            "major": "행정학과, 법학과, 정치외교학과",
            "personality": "성실하고 계획적이며 맡은 일을 꾸준하게 수행하는 사람",
            "salary": "약 3,000만~6,000만 원"
        }
    ],

    "ISFJ": [
        {
            "job": "간호사",
            "major": "간호학과",
            "personality": "배려심이 많고 세심하며 다른 사람을 돕는 것을 좋아하는 사람",
            "salary": "약 3,500만~5,500만 원"
        },
        {
            "job": "초등학교 교사",
            "major": "초등교육과",
            "personality": "책임감이 있고 친절하며 아이들의 성장을 돕는 사람",
            "salary": "약 4,000만~7,000만 원"
        }
    ],

    "INFJ": [
        {
            "job": "상담심리사",
            "major": "심리학과, 상담심리학과",
            "personality": "공감 능력이 뛰어나고 다른 사람의 마음을 이해하는 사람",
            "salary": "약 3,000만~5,000만 원"
        },
        {
            "job": "작가",
            "major": "국어국문학과, 문예창작학과",
            "personality": "상상력이 풍부하고 자신의 생각을 글로 표현하는 사람",
            "salary": "약 2,500만~5,000만 원"
        }
    ],

    "INTJ": [
        {
            "job": "소프트웨어 개발자",
            "major": "컴퓨터공학과, 소프트웨어학과",
            "personality": "논리적이고 분석적이며 문제 해결을 즐기는 사람",
            "salary": "약 4,000만~8,000만 원"
        },
        {
            "job": "연구원",
            "major": "자연과학계열, 공학계열",
            "personality": "탐구심이 강하고 독립적으로 깊이 연구하는 사람",
            "salary": "약 3,500만~7,000만 원"
        }
    ],

    "ISTP": [
        {
            "job": "기계공학 기술자",
            "major": "기계공학과, 메카트로닉스공학과",
            "personality": "실용적이고 관찰력이 뛰어나며 기계에 관심이 많은 사람",
            "salary": "약 3,500만~6,000만 원"
        },
        {
            "job": "항공정비사",
            "major": "항공정비학과, 항공기계공학과",
            "personality": "침착하고 세밀하며 기술적인 문제를 해결하는 사람",
            "salary": "약 3,500만~6,000만 원"
        }
    ],

    "ISFP": [
        {
            "job": "그래픽 디자이너",
            "major": "시각디자인학과, 산업디자인학과",
            "personality": "감각적이고 창의적이며 자신만의 아이디어를 표현하는 사람",
            "salary": "약 2,800만~5,000만 원"
        },
        {
            "job": "수의사",
            "major": "수의학과",
            "personality": "생명을 소중히 여기고 섬세하며 책임감이 있는 사람",
            "salary": "약 4,000만~7,000만 원"
        }
    ],

    "INFP": [
        {
            "job": "웹툰 작가",
            "major": "만화애니메이션학과, 웹툰학과",
            "personality": "상상력이 풍부하고 창의적이며 이야기를 만드는 사람",
            "salary": "약 2,500만~5,000만 원"
        },
        {
            "job": "사회복지사",
            "major": "사회복지학과",
            "personality": "공감 능력이 뛰어나고 따뜻한 마음으로 사람을 돕는 사람",
            "salary": "약 2,800만~4,500만 원"
        }
    ],

    "INTP": [
        {
            "job": "데이터 분석가",
            "major": "통계학과, 데이터사이언스학과",
            "personality": "호기심이 많고 논리적으로 자료를 분석하는 사람",
            "salary": "약 3,500만~7,000만 원"
        },
        {
            "job": "AI 연구원",
            "major": "인공지능학과, 컴퓨터공학과",
            "personality": "새로운 기술을 탐구하고 복잡한 문제를 해결하는 사람",
            "salary": "약 4,000만~9,000만 원"
        }
    ],

    "ESTP": [
        {
            "job": "마케팅 전문가",
            "major": "경영학과, 광고홍보학과",
            "personality": "활동적이고 도전적이며 빠르게 상황을 파악하는 사람",
            "salary": "약 3,000만~6,000만 원"
        },
        {
            "job": "스포츠 지도자",
            "major": "체육학과, 스포츠과학과",
            "personality": "에너지가 넘치고 사람들과 함께 활동하는 것을 좋아하는 사람",
            "salary": "약 2,800만~5,000만 원"
        }
    ],

    "ESFP": [
        {
            "job": "콘텐츠 크리에이터",
            "major": "미디어학과, 방송영상학과",
            "personality": "사교적이고 표현력이 풍부하며 소통을 즐기는 사람",
            "salary": "약 2,500만~6,000만 원"
        },
        {
            "job": "이벤트 기획자",
            "major": "이벤트학과, 광고홍보학과",
            "personality": "밝고 적극적이며 즐거운 행사를 기획하는 사람",
            "salary": "약 2,800만~5,000만 원"
        }
    ],

    "ENFP": [
        {
            "job": "광고 기획자",
            "major": "광고홍보학과, 미디어학과",
            "personality": "아이디어가 풍부하고 새로운 시도를 즐기는 사람",
            "salary": "약 3,000만~6,000만 원"
        },
        {
            "job": "진로 상담 전문가",
            "major": "교육학과, 상담심리학과",
            "personality": "사람들의 가능성을 발견하고 응원하는 사람",
            "salary": "약 3,000만~5,000만 원"
        }
    ],

    "ENTP": [
        {
            "job": "창업가",
            "major": "경영학과, 창업학과",
            "personality": "도전 정신이 강하고 새로운 아이디어를 발전시키는 사람",
            "salary": "개인과 사업 성과에 따라 차이가 큼"
        },
        {
            "job": "변호사",
            "major": "법학과, 법학전문대학원",
            "personality": "논리적이고 토론을 즐기며 다양한 관점으로 생각하는 사람",
            "salary": "약 5,000만~1억 원 이상"
        }
    ],

    "ESTJ": [
        {
            "job": "경영 관리자",
            "major": "경영학과, 경제학과",
            "personality": "조직적이고 추진력이 있으며 목표 달성을 중요하게 생각하는 사람",
            "salary": "약 4,000만~8,000만 원"
        },
        {
            "job": "경찰관",
            "major": "경찰행정학과, 법학과",
            "personality": "책임감이 강하고 원칙을 지키며 판단력이 뛰어난 사람",
            "salary": "약 3,500만~6,000만 원"
        }
    ],

    "ESFJ": [
        {
            "job": "인사 담당자",
            "major": "경영학과, 행정학과",
            "personality": "친화력이 좋고 협력과 조직 관리를 중요하게 생각하는 사람",
            "salary": "약 3,000만~6,000만 원"
        },
        {
            "job": "유치원 교사",
            "major": "유아교육과",
            "personality": "친절하고 사교적이며 아이들과 소통하는 것을 좋아하는 사람",
            "salary": "약 2,800만~4,500만 원"
        }
    ],

    "ENFJ": [
        {
            "job": "교사",
            "major": "교육학과, 사범대학 관련 학과",
            "personality": "리더십이 있고 다른 사람의 성장을 돕는 것을 좋아하는 사람",
            "salary": "약 4,000만~7,000만 원"
        },
        {
            "job": "기업 교육 전문가",
            "major": "교육학과, 경영학과",
            "personality": "소통 능력이 뛰어나고 사람들의 역량을 키우는 데 관심이 많은 사람",
            "salary": "약 3,500만~6,000만 원"
        }
    ],

    "ENTJ": [
        {
            "job": "기업 경영인",
            "major": "경영학과, 경제학과",
            "personality": "목표 지향적이고 결단력이 있으며 조직을 이끄는 사람",
            "salary": "개인과 기업 성과에 따라 차이가 큼"
        },
        {
            "job": "변리사",
            "major": "공학계열, 자연과학계열, 법학 관련 분야",
            "personality": "분석력이 뛰어나고 전문 지식으로 문제를 해결하는 사람",
            "salary": "약 5,000만~9,000만 원 이상"
        }
    ]
}

# =====================================
# 4. 메인 화면
# =====================================

st.markdown(
    '<div class="main-title">🌈 MBTI 진로 탐험소</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="sub-title">'
    '나의 성격에 어울리는 미래 직업을 찾아보자! 🚀'
    '</div>',
    unsafe_allow_html=True
)

st.write("")

st.info(
    "💡 MBTI는 진로를 결정하는 정답이 아니라 "
    "나의 관심과 성향을 알아보는 참고 자료예요!"
)

# =====================================
# 5. MBTI 선택
# =====================================

st.markdown("### 🧩 나의 MBTI를 선택해 줘!")

mbti_list = [
    "ISTJ", "ISFJ", "INFJ", "INTJ",
    "ISTP", "ISFP", "INFP", "INTP",
    "ESTP", "ESFP", "ENFP", "ENTP",
    "ESTJ", "ESFJ", "ENFJ", "ENTJ"
]

selected_mbti = st.selectbox(
    "나의 MBTI 유형은?",
    mbti_list,
    index=None,
    placeholder="MBTI를 선택해 주세요!"
)

# =====================================
# 6. 진로 추천 결과
# =====================================

if selected_mbti:

    st.divider()

    st.markdown(
        f"## 🎉 {selected_mbti} 유형의 추천 진로"
    )

    st.success(
        "당신에게 어울릴 수 있는 진로 2가지를 소개할게요!"
    )

    jobs = career_data[selected_mbti]

    col1, col2 = st.columns(2)

    for index, job_info in enumerate(jobs):

        if index == 0:
            current_col = col1
        else:
            current_col = col2

        with current_col:

            emoji = "🌟" if index == 0 else "✨"

            st.markdown(
                f"""
                <div class="job-card">

                    <div class="job-name">
                        {emoji} {job_info['job']}
                    </div>

                    <br>

                    <div class="info">

                        🎓 <b>추천 학과</b><br>
                        {job_info['major']}

                        <br><br>

                        💗 <b>어울리는 성격 및 역량</b><br>
                        {job_info['personality']}

                        <br><br>

                        💰 <b>예상 연봉</b><br>
                        {job_info['salary']}

                    </div>
                </div>
                """,
                unsafe_allow_html=True
            )

    # =================================
    # 7. 추가 진로 탐색 안내
    # =================================

    st.divider()

    st.markdown("### 💬 나만의 진로를 찾아가는 방법!")

    st.markdown("""
    🚀 **진로 탐색 TIP**

    1. 🔍 내가 좋아하는 과목과 활동을 찾아보기
    2. 📚 추천 직업에 필요한 학과와 공부 알아보기
    3. 🎯 관심 있는 직업을 직접 체험해 보기
    4. 💡 나의 장점과 흥미를 함께 고려하기

    **나의 미래는 내가 만들어 가는 거야! 💕**
    """)

else:

    st.warning(
        "👇 위에서 MBTI 유형을 선택하면 "
        "추천 진로가 나타나요!"
    )

# =====================================
# 8. 하단 안내
# =====================================

st.divider()

st.markdown("""
### 📌 꼭 확인해 주세요!

- 연봉은 국내 직업별 임금 수준을 참고한 대략적인 참고 범위입니다.
- 실제 연봉은 경력, 회사 규모, 지역 등에 따라 달라질 수 있습니다.
- MBTI와 특정 직업의 적합성이 과학적으로 확정된 것은 아닙니다.
- 자신의 흥미, 적성, 가치관도 함께 고려해 주세요.
""")

st.markdown("""
🔎 더 알아보기

- [커리어넷 직업정보](https://www.career.go.kr/)
- [한국고용정보원 워크피디아](https://www.wagework.go.kr/)
""")

st.markdown(
    '<div class="footer">'
    '🌈 MBTI 진로 탐험소 | 나의 가능성을 발견하는 시간'
    '</div>',
    unsafe_allow_html=True
)
