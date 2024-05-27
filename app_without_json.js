import React, { useState } from 'react';
import Chat from './Chat';
import ChatInput from './ChatInput';
import ChatHistory from './ChatHistory';
import './App.css';

const App = () => {
  const [messages, setMessages] = useState([]);
  const [history, setHistory] = useState([]);
  const [showHistory, setShowHistory] = useState(false);
  const [loading, setLoading] = useState(false);

  const handleNewChat = () => {
    setMessages([]);
  };

  const handleSendMessage = async (message) => {
    setLoading(true);
    try {
      const response = await fetch('/api/chat', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/x-www-form-urlencoded', // Use form-encoded format
        },
        body: new URLSearchParams({ msg: message }), // Form-encoded data
      });
      const data = await response.text(); // Assuming the response is plain text
      setMessages([...messages, { user: message, bot: data }]);
      setHistory([...history, { user: message, bot: data }]);
    } catch (error) {
      console.error("Error sending message:", error);
    } finally {
      setLoading(false);
    }
  };

  const handleShowHistory = () => {
    setShowHistory(!showHistory);
  };

  return (
    <div className="app">
      <div className="header">
        <h1 className="app-title">PIA</h1>
      </div>
      <div className="buttons">
        <button className="new-chat-button" onClick={handleNewChat}>+ New Chat</button>
        <button className="history-button" onClick={handleShowHistory}>Chat History</button>
      </div>
      {showHistory ? (
        <ChatHistory history={history} />
      ) : (
        <>
          <Chat messages={messages} loading={loading} />
          <ChatInput onSendMessage={handleSendMessage} />
        </>
      )}
    </div>
  );
};

export default App;
