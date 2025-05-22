+++
date = '2025-05-22T17:00:21+09:00'
title = 'Github pages 를 이용한 blog Setting (hugo + hugo-book theme + giscus'
weight = 3
+++
# **Github pages 를 이용한 blog Setting (hugo + hugo-book theme + giscus**

## **0. blog host 방식 선택**

여러 블로그 host방식을 고려하였으나, 유지보수가 손이 덜가며, 오랫동안 hosting이 되는것이 우선 순위였고.

**Github pages**를 통한 호스팅 방법을 선택하였다.

그렇다면, static page를 generation을 해주는 framework로 jekyll 와 hugo를 고민하였고 
ruby 보다 go로 이루어진 hugo를 믿기로 하였다.

다른 blog들이 생성한 페이지를 주로 참조하였으며, 이 포스팅은 해당 글들의 엮음에 불과한점을 참조 부탁한다.

## **1. Setup git, hugo**

개발환경은 window 11 (Intel processor) 이므로 이를 기반으로 작성되었다.

### **git 설치**

https://git-scm.com/downloads 사이트에서  `Git for Windows/x64 Setup.` 항목을 클릭하여 github을 설치하였다.
각 설정은 default를 그대로 사용하였다.

### **hugo 설치**

https://github.com/gohugoio/hugo/releases
에서 본인에 맞는 최신 버전을 받는다.  (hugo_extended_0.147.4_windows-amd64.zip)

후에 언급하겠지만 `hugo-book` theme 의 경우 hugp-extended 버전을 받아야했다.

hugo 명령어를 치는 경우가 많으므로, 환경변수에 등록하였다. (되도록 path 파싱에 에러가 없도록 폴더들을 영어로 구성하였다.)

### **github repository 생성**

https://minyeamer.github.io/blog/hugo-blog-1/ 에서 언급된것 같이 하나의 repository를 branch로 나누어서 submodule로 활용하는 방안을 채택했다.

1. `<USERNAME>.github.io` 인 repository를 하나 생성한다. <br>
UserName이 Jon 이라면 다음과 같이 생성될 것이다. `Jon/Jon.github.io`

2. hugo를 통한 기본 뼈대 만들기. 여기서 폴더명을 생성한 repository로 맞춘다.
```sh
hugo new site <USERNAME>.github.io
cd <USERNAME>.github.io
git init
git add .
git commit -m "feat: new site"
git branch -M main
git remote add origin https://github.com/<USERNAME>/<USERNAME>.github.io.git
git push -u origin main
```
3. branch 만들기
```sh
git branch gh-pages main
git checkout gh-pages
git push origin gh-pages
git checkout main
```
4. gh-pages를 submoudle로 연동하기
```sh
rm -rf public
git submodule add -b gh-pages https://github.com/<USERNAME>/<USERNAME>.github.io.git public 
git add public
git add .gitmodules
git commit -m "feat: add submodule for github pages"
git push
```

### **hugo Theme : hugo-book**

https://themes.gohugo.io/ 에서 여러 테마가 선택이 가능하고, 가벼워보이는 hugo-book theme 를 채택하였다

https://github.com/alex-shpak/hugo-book 에 Read.md를 참조하였다.

```sh
git submodule add https://github.com/alex-shpak/hugo-book themes/hugo-book
git add .
git commit -m "feat: import hugo theme"
```

root path 에 있는 hugo.toml 파일에 아래 내용을 삽입한다.
```
baseURL = 'https://<USERNAME>.github.io'
languageCode = 'ko-kr'
title = "< title what you want>"
theme = 'hugo-book'
```


### **Github 설정**

Github -> <USERNAME>.github.io -> Settings -> Pages -> Branch -> `gh-pages` 로 변경

## **2. 블로그 배포 방법**

### **hugo static page 생성**

아래 명령어로 `<root-path>/content/` path에 firstPost.md 파일이 생성된다
```sh
hugo new firstPost.md
```

아래 커맨드를 이용하면 현재 static 페이지를 `127.0.0.1:1313` 에서 확인이 가능하다

```sh
hugo server -D
```
아래와 같이 draft가 true가 있다면 디버깅은 가능하지만 최종 산출물에서 빠지게 된다. 따라서 나중에 빼놓지 말고 
`draft = true` 문구를 지우던 false로 변경한다
```
// firstPost.md
+++
title = 'firstPost'
draft = true
+++
```

아래 명령어를 통해서 public/ 폴더 밑에 산출물들을 생성한다.
```sh
hugo
```

아래 명령어를 통해서 github repository에 산출물들을 배포한다. 이 산출물들은 gitub action 에 의해서 자동으로 hosting되도록 처리된다.
```sh
cd public
git add .
git commit -m "<comment what you want to add>"
git push origin gh-pages
cd ..
git add .
git commit -m "<comment what you want to add>"
git push origin main
```

## **3. 추가 hugo 세팅팁**
### **recent-post 용 home 만들기**

아래와 같이 폴더 및 기본 md 파일을 구성하였고.

```
content
├── _index.md
├── Content-Category1
│   ├── _index.md
│   └── article1.md
└── Content-Category2
     ├── _index.md
     └── post1.md
```

`conent/_index.md` 파일은 다음과 같이 작성하였다.
{{% hint info %}}
"recent-posts" 에서 " 을 삭제
{{% /hint %}}

```markdown 
---
title: "Home"
comments: false
---
# **Recent Posts**
{{ {{ "recent-posts" }} }}

```

`layouts/shortcodes/recent-posts.html` 을 생성하여 아래와 같이 작성
```html
<style>
.summary-box {
  background: #f8f9fa;
  padding: 1rem;
  border-radius: 8px;
  transition: background 0.3s ease;
  max-height: 14rem;         /* 높이 제한 */
  overflow: hidden;         /* 넘치는 내용 숨김 */
  display: -webkit-box;     
  -webkit-line-clamp: 6;    /* 최대 줄 수 (예: 6줄) */
  -webkit-box-orient: vertical;
  text-overflow: ellipsis;  /* ... 처리 */
}

.summary-box h1,
.summary-box h2,
.summary-box h3,
.summary-box h4,
.summary-box h5,
.summary-box h6,
.summary-box p {
  margin: 0 0 1rem 0;
}

.summary-box:hover {
  background: #e9ecef;
}
</style>

<ul>
{{ range (first 5 (sort .Site.RegularPages "Date" "desc")) }}
  <li style="margin-bottom: 2rem;">
    <a href="{{ .RelPermalink }}">
      <strong style="font-size: 1.25rem;">{{ .Title }}</strong>
    </a><br/>

    {{ with .Params.subtitle }}
      <small style="color: #888;">{{ . }}</small><br/>
    {{ end }}

    <small>{{ .Date.Format "2006-01-02 15:04:05 MST" }}</small>

    <a href="{{ .RelPermalink }}" style="text-decoration: none; color: inherit;">
      <div class="summary-box"> <!-- delete anchor in title-->
        {{ .Summary | replaceRE "<a class=\"anchor\" href=\"#.*?\">#</a>" ""  | safeHTML | truncate 400}}
      </div>
    </a>

  </li>
{{ end }}
</ul>
```
### **MarkDown code block copy 버튼 만들기**
`layouts/partials/docs/body.html` 을 생성하여 아래와 같이 작성
```javaScript
<script>
  document.addEventListener('DOMContentLoaded', () => {
    document.querySelectorAll('pre > code[class^="language-"]').forEach(codeBlock => {
      const pre = codeBlock.parentElement;

      // 코드블럭의 복사 버튼  생성
      const button = document.createElement('button');
      button.innerText = '📋';
      button.title = 'Copy code';
      button.style = `
        position: absolute;
        top: 0.5em;
        right: 0.5em;
        padding: 2px 6px;
        font-size: 0.8rem;
        background: #f5f5f5;
        border: 1px solid #ccc;
        border-radius: 4px;
        cursor: pointer;
        z-index: 10;
      `;

      // 복사 동작 정의
      button.addEventListener('click', () => {
        navigator.clipboard.writeText(codeBlock.innerText);
        button.innerText = '✔';
        setTimeout(() => button.innerText = '📋', 1000);
      });

      // 스타일 적용
      pre.style.position = 'relative';
      pre.appendChild(button);
    });
  });
</script>
```

## **4. giscus 연동하기**

### **giscus github에 설치 및 discuss 활성화**
https://github.com/apps/giscus 에서 `giscus` install

github page repository 만 선택

Setting -> Discusss 활성화

Discussions 에서 좌측 Categories 선택하여 새 카테고리 만들기

`Category Name` 와 `Description`을 각각 입력하고 `Discussion Format` 을 `Announcement` 선택

### **hugo 에서 보이게 설정**
https://giscus.app/ko 페이지에 접속하여 
1. <USERNAME>/<USERNAME>.github.io 로 저장소 입력
2. Discussion 제목이 페이지 경로 포함 선택
3. 페이지에 생성된 script를 복사 (아래와 유사한 형식)
```javascript
<script src="https://giscus.app/client.js"
        data-repo="<USERNAME>/<USERNAME<.github.io"
        data-repo-id="< String >"
        data-category="Comment"
        data-category-id="< String >"
        data-mapping="pathname"
        data-strict="0"
        data-reactions-enabled="1"
        data-emit-metadata="0"
        data-input-position="bottom"
        data-theme="preferred_color_scheme"
        data-lang="ko"
        crossorigin="anonymous"
        async>
</script>
```

4. 이를 `layouts/partials/comments.html에 아래처럼 붙여넣기
```javascript
{{ if and (.IsPage) (not .IsHome) (not (eq .Params.comments false)) }}
<script src="https://giscus.app/client.js"
        data-repo="<USERNAME>/<USERNAME<.github.io"
        data-repo-id="< String >"
        data-category="Comment"
        data-category-id="< String >"
        data-mapping="pathname"
        data-strict="0"
        data-reactions-enabled="1"
        data-emit-metadata="0"
        data-input-position="bottom"
        data-theme="preferred_color_scheme"
        data-lang="ko"
        crossorigin="anonymous"
        async>
</script>
{{ end }}
```
{{% hint info %}}
comments = false를 넣은 페이지들은 github 코멘트가 안보임
{{% /hint %}}

## **5. 검색 연동**

### **구굴 서치 콘솔**

1. 소유권 확인
https://search.google.com/search-console/about 접속하여 url `<USERNAME>.github.io` 을 입력

google_<blabla>.html 을 다운로드 후에 `public/` 에 위치시킴

2. sitemap 만들기 hugo.toml 에 아래와 같이 삽입 -> sitemap.xml이 생성된것을 확인
```toml
enableRobotsTXT = true
[sitemap]
# always, hourly daily, weekly, monthly, yearly, never
  changefreq = "always"
  filename = "sitemap.xml"
  priority = 0.5
```

3. `public/robot.txt` 파일을 수정
```yaml
User-agent: *
Allow: /
Sitemap: {{ '/sitemap.xml' | relative_url | prepend: site.url }}
```
5. sitemap.xml을 서치 콘솔에 등록

### **네이버 서치 어드바이져**
// TODO: google search console 및 네이버 서치 어드바이져

## **6. tag 설정**
// TODO: 글마다 tag 설정 가능하게 하기

## **참조**
https://ialy1595.github.io/post/blog-construct-1/

https://minyeamer.github.io/blog/hugo-blog-1/

https://d5br5.dev/blog/nextjs_blog/giscus

https://velog.io/@eona1301/Github-Blog-%EA%B2%80%EC%83%89%EC%B0%BD-%EB%85%B8%EC%B6%9C%EC%8B%9C%ED%82%A4%EA%B8%B0