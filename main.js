// Function to show SQL Query overlay
function showSQL() {
    $.ajax({
        type: "GET",
        url: "/sql",
        data: { query: $("#text").val() }
    }).done(function(data) {
        $("#sqlOverlay .overlay-content").html('<pre>' + JSON.stringify(data.sql_query, null, 2) + '</pre>');
        $("#sqlOverlay").show();
    });
}

// Function to show Visual overlay
function showVisual() {
    $.ajax({
        type: "GET",
        url: "/visual",
        data: { query: $("#text").val() }
    }).done(function(data) {
        $("#visualOverlay .overlay-content").html(data.plot);
        $("#visualOverlay").show();
    });
}

// Function to close overlays
function closeOverlay() {
    $("#sqlOverlay").hide();
    $("#visualOverlay").hide();
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
            url: "/api/chat",
        }).done(function(data) {
            const date_done = new Date();
            const hour_done = date_done.getHours();
            const minute_done = (date_done.getMinutes() < 10 ? '0' : '') + date_done.getMinutes();
            const str_time_done = hour_done + ":" + minute_done;

            $("#typing").remove();
            var botHtml = '<div class="d-flex justify-content-start mb-4"><div class="msg_container"><p id="h7_send">' + data.response + '</p><span class="msg_time">' + str_time_done + '</span></div></div>';
            $("#messageFormeight").append(botHtml);
            $("#messageFormeight").animate({ scrollTop: $("#messageFormeight")[0].scrollHeight }, "slow");

            // Show SQL Query and Visual buttons if applicable
            if (data.sql_query || data.plot) {
                $("#showSQL").show();
                $("#showVisual").show();
            } else {
                $("#showSQL").hide();
                $("#showVisual").hide();
            }
        });
    });

    // Initially hide SQL Query and Visual buttons
    $("#showSQL").hide();
    $('#showVisual').hide();

    // Show SQL Query button event
    $("#showSQL").on("click", showSQL);

    // Show Visual button event
    $("#showVisual").on("click", showVisual);
});

