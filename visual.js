function VisualButton({ plotData }) {
  const [show, setShow] = useState(false);

  return (
    <div>
      <button onClick={() => setShow(!show)}>View Visual</button>
      {show && (
        <div className="overlay">
          <div className="overlay-content">
            {plotData ? (
              <Plotly.Plot
                data={plotData.data}
                layout={plotData.layout}
              />
            ) : (
              <p>No data to visualize.</p>
            )}
          </div>
        </div>
      )}
    </div>
  );
}
