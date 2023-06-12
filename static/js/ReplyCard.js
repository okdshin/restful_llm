export default {
    props: {
        prompt: String,
        generated_text: String,
    },
    template: `
    <div class="card mt-3">
        <div class="col">
            <div class="card-body">
                <p class="card-text">prompt: "{{prompt}}"</p>
                <h2 class="card-text">"{{generated_text}}"</h2>
            </div>
        </div>
    </div>
    `
}
