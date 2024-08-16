$(document).ready(function() {
    // Function to submit user message
    $("#messageArea").on("submit", function(event) {
        event.preventDefault(); // Prevent default form submission

        const date = new Date();
        const hour = date.getHours();
        const minute = (date.getMinutes() < 10 ? '0' : '') + date.getMinutes();
        const str_time = hour + ":" + minute;

        var rawText = $("#text").val();
        var userHtml = '<div class="d-flex justify-content-end mb-4"><div class="msg_container_send"><p id="h7">' + rawText + '</p><span class="msg_time_send">' + str_time + '</span></div></div>';
        $("#text").val("");
        $("#messageFormeight").append(userHtml);
        $("#messageFormeight").animate({ scrollTop: $("#messageFormeight")[0].scrollHeight }, "slow");

        var loadingHTML = '<div class="d-flex justify-content-start mb-4"><div class="typing-indicator" id="typing"><span></span><span></span><span></span></div></div>';
        $("#messageFormeight").append($.parseHTML(loadingHTML));

        $.ajax({
            data: { msg: rawText },
            type: "POST",
            url: "/get",
        }).done(function(data) {
            const date_done = new Date();
            const hour_done = date_done.getHours();
            const minute_done = (date_done.getMinutes() < 10 ? '0' : '') + date_done.getMinutes();
            const str_time_done = hour_done + ":" + minute_done;

            $("#typing").remove();
            var botHtml = '<div class="d-flex justify-content-start mb-4"><div class="msg_container"><p id="h7_send">' + data.response + '</p><span class="msg_time">' + str_time_done + '</span></div></div>';
            $("#messageFormeight").append(botHtml);

            // Append Show SQL and Show Visual buttons
            var buttonsHtml = `
                <div class="d-flex justify-content-start mb-4">
                    <button id="showSQL" class="sql-btn">Show SQL Query</button>
                    <button id="showVisual" class="visual-btn">Show Visual</button>
                </div>
            `;
            $("#messageFormeight").append(buttonsHtml);

            $("#messageFormeight").animate({ scrollTop: $("#messageFormeight")[0].scrollHeight }, "slow");

            // Add event listeners for the buttons
            $("#showSQL").on("click", function() {
                showSQL();
            });

            $("#showVisual").on("click", function() {
                showVisual();
            });
        });
    });

    // Initially hide sql_query and visual button
    $("#showSQL").hide();
    $('#showVisual').hide();
});
