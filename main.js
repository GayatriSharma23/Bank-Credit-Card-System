// Functions to show and close overlays
function showSQL() {
    document.getElementById("sqlOverlay").style.display = "block";
}

function showVisual() {
    document.getElementById("visualOverlay").style.display = "block";
}

function closeOverlay() {
    document.getElementById('sqlOverlay').style.display = 'none';
    document.getElementById('visualOverlay').style.display = 'none';
}

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

            // Add buttons if SQL query or visual data exists
            if (data.sql_query || data.plot) {
                var buttonHtml = '<div class="button-container" style="margin-top: 10px;">';
                if (data.sql_query) {
                    buttonHtml += '<button class="sql-btn" id="showSQL" style="display: block;">Show SQL Query</button>';
                }
                if (data.plot) {
                    buttonHtml += '<button class="visual-btn" id="showVisual" style="display: block;">Show Visual</button>';
                }
                buttonHtml += '</div>';

                // Append the buttons to the same container as the bot's response
                $("#messageFormeight").append(buttonHtml);
            }

            $("#messageFormeight").animate({ scrollTop: $("#messageFormeight")[0].scrollHeight }, "slow");

            // Add event listeners for buttons after they are appended to the DOM
            $('#showSQL').on('click', function() {
                $.ajax({
                    type: "GET",
                    url: "/sql",
                }).done(function(data) {
                    showOverlay(data);
                });
            });

            $('#showVisual').on('click', function() {
                $.ajax({
                    type: "GET",
                    url: "/visual",
                }).done(function(data) {
                    showOverlay(data);
                });
            });
        });
    });

    // Function to show overlay with content
    function showOverlay(content) {
        $("#overlayContent").html(content);
        $("#overlay").show();
    }

    // Function to close overlay
    function closeOverlay() {
        $("#overlay").hide();
    }
});
