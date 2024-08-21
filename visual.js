import React, { useState } from "react";
import Plot from 'react-plotly.js';

function VisualButton({ plotData }) {
    const [show, setShow] = useState(false);
    const toggleOverlay = () => setShow(!show);

    // Function to validate plotData structure
    const isValidPlotData = (plotData) => {
        if (!plotData || !Array.isArray(plotData.data)) {
            console.warn("plotData.data is not an array or plotData is undefined");
            return false;
        }

        for (const trace of plotData.data) {
            if (typeof trace !== 'object' || !trace.hasOwnProperty('x') || !trace.hasOwnProperty('y') || !trace.hasOwnProperty('type')) {
                console.warn("Invalid trace data:", trace);
                return false;
            }
        }

        return true;
    };

    console.log("Visual Button plot:", plotData);
    console.log("Visual Button type:", typeof plotData);

    return (
        <div>
            <button className="visual-btn" onClick={toggleOverlay}>
                Show Visual
            </button>
            {show && (
                <div className="overlay">
                    <div className="overlay-content">
                        {isValidPlotData(plotData) ? (
                            <Plot
                                data={plotData.data}  // Ensure data is an array
                                layout={plotData.layout || {width: 500, height: 400}} // Ensure layout is passed
                            />
                        ) : (
                            <p>No data to visualize</p>
                        )}
                        <button onClick={toggleOverlay}>Close</button>
                    </div>
                </div>
            )}
        </div>
    );
}

export default VisualButton;
