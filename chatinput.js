import React, { useState } from 'react';

const ChatInput = ({ onSendMessage }) => {
  const [input, setInput] = useState('');

  const handleSend = () => {
    onSendMessage(input);
    setInput('');
  };

  return (
    <div className="chat-input">
      <input
        type="text"
        value={input}
        onChange={(e) => setInput(e.target.value)}
        placeholder="Type your query here..."
      />
      <button onClick={handleSend}>Send</button>
    </div>
  );
};

export default ChatInput;
