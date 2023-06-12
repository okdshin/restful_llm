export default {
    props: {},
    data() {
        return {
            query_and_reply_list: [],
            is_processing: false,

            max_length_limit: 128,

            max_length: 128,
            do_sample: true,
            top_k: 50,
            top_p: 0.95,
            temperature: 1,
            prompt: 'In a shocking finding, scientists discovered a herd of unicorns living in a remote, previously unexplored valley in the Andes Mountains. Even more surprising to the researchers was the fact that the unicorns spoke perfect English.',
        }
    },
    created : function(){
        let self = this;
        axios
            .get('/api/v1/default_config/')
            .then(function(response) {
                self.max_length_limit = response.data.max_length;
            });
    },
    methods: {
        submitUserPrompt() {
            let self = this;
            self.is_processing = true;
            let options = {
                    max_length: self.max_length,
                    do_sample: self.do_sample,
                    top_k: self.top_k,
                    top_p: self.top_p,
                    temperature: self.temperature,
                    input_text: self.prompt,
                };
            axios
                .post('/api/v1/generate_text/', options)
                .then(function(response) {
                    self.query_and_reply_list.push({
                        options: options,
                        generated_text: response.data.text
                    });
                })
                .finally(function() {
                    self.is_processing = false;
                });
        }
    },
    template: `
    <div class="col">
        <form>
            <div class="form-group">
                <label>
                    Prompt
                </label>
                <textarea v-model="prompt" class="form-control" id="prompt" rows="3" placeholder="input prompt here"></textarea>
            </div>
        </form>
    </div>
    <div class="col">
        <form>
            <div class="form-group">
                <label for="max_length" class="form-label">max_length={{max_length}}</label>
                <input v-model="max_length" type="range" class="form-range" id="width" min="0" max=2048 step="8">
            </div>
            <div class="form-group">
                <input type="checkbox" id="do_sample" v-model="do_sample">
                do_sample
            </div>
            <div class="form-group">
                <label for="top_k" class="form-label">top_k={{top_k}}</label>
                <input v-model="top_k" type="range" class="form-range" id="top_k" min="10" max="100" step="10">
            </div>

            <div class="form-group">
                <label for="top_p" class="form-label">top_p={{top_p}}</label>
                <input v-model="top_p" type="range" class="form-range" id="top_p" min="0.05" max="1.0" step="0.05">
            </div>

            <div class="form-group">
                <label for="temperature" class="form-label">temperature={{temperature}}</label>
                <input v-model="temperature" type="range" class="form-range" id="temperature" min="0.0" max="3.0" step="0.1">
            </div>
        </form>
    </div>
    <form>
        <button @click="submitUserPrompt" :disabled="is_processing || (prompt.length === 0)" type="button" class="mt-1 col-12 btn btn-primary">
            <template v-if="is_processing">
                <div class="spinner-border" role="status" style="width:1rem; height: 1rem"></div>
                Processing...
            </template>
            <template v-else>
                <i class="bi bi-arrow-down fs-8"></i>
                Generate
            </template>
        </button>
    </form>

    <template v-for="item in query_and_reply_list.slice().reverse()">
        <li class="list-group-item">
            <reply-card
                :prompt="item.options.input_text"
                :generated_text="item.generated_text"
            />
        </li>
    </template>
    `
}
