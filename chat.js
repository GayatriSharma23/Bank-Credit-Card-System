import React from 'react';

const Chat = ({ messages, loading, sqlQuery, onViewSQL, visualData, onViewVisual }) => {
  return (
    <div className="chat">
      {messages.map((msg, index) => (
        <div key={index} className={msg.type === 'user' ? 'user-message' : 'bot-message'}>
          <div className="message">
            {msg.type === 'user' && (
              <div className="user-message">
                <div className="user-content">{msg.content}</div>
              </div>
            )}
            {msg.type === 'bot' && (
              <div className="bot-message">
                <div className="bot-icon"></div>
                <div className="bot-content">
                  {msg.content}
                  {sqlQuery && (
                    <button className="view-sql-button" onClick={onViewSQL}>View SQL Query</button>
                  )}
                  {visualData && (
                    <button className="view-visual-button" onClick={onViewVisual}>View Visual</button>
                  )}
                </div>
              </div>
            )}
          </div>
        </div>
      ))}
      {loading && (
        <div className="loading-dots">
          <span>.</span>
          <span>.</span>
          <span>.</span>
        </div>
      )}
    </div>
  );
};

export default Chat;

