import SentimentAnalyzer from "./Components/SentimentAnalyzer";
import bgVideo from './assets/emoji.mp4';
// import bgImage from './assets/sentiment.jpg'

function App() {
  console.log(bgVideo);
  return (
    <div className="relative min-h-screen">
      <video
        autoPlay
        loop
        muted
        className="absolute top-0 left-0 w-full h-full object-cover"
      >
        <source src={bgVideo} type="video/mp4" />
        Your browser does not support the video tag.
      </video>
      <div className="relative z-10 bg-opacity-70 bg-white">
        <SentimentAnalyzer />
      </div>
    </div>
  );
}

export default App;
