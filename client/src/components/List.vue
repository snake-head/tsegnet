<template>
    <div>
        <el-table
            ref="singleTable"
            :data="tableData"
            highlight-current-row
            @current-change="handleCurrentChange"
            style="width: 100%">
            <el-table-column
            type="index"
            width="50">
            </el-table-column>
            <el-table-column
            property="filename"
            label="文件名"
            width="200">
            </el-table-column>
            <el-table-column
            property="tag"
            label="上颌/下颌"
            width="120"
            :filters="[{ text: '上颌', value: '上颌' }, { text: '下颌', value: '下颌' }]"
            :filter-method="filterTag"
            filter-placement="bottom-end">
            <template v-slot="scope">
                <el-tag
                    :type="scope.row.tag === '上颌' ? '' : 'success'"
                    disable-transitions>{{scope.row.tag}}</el-tag>
            </template>
            </el-table-column>
        </el-table>
        <el-button id="importButton" type="primary" @click="importTooth(currentRow)">导入</el-button>
    </div>
</template>

<script>
  
    import axios from 'axios'
    export default {
        name:'List',
        data() {
          return {
            tableData: [],
            currentRow: null,
          }
        },

        methods: {
        setCurrent(row) {
            this.$refs.singleTable.setCurrentRow(row);
        },
        handleCurrentChange(val) {
            this.currentRow = val;
        },
        filterTag(value, row) {
            return row.tag === value;
        },
        importTooth(currentRow){
          if (currentRow === null){
            alert('未选择文件')
          }
          else{
            if (currentRow['tag'] === '上颌'){
              //若vtkContext中有内容，且模型地址发生改变，则清空vtkContext和centroidsPred中的内容
              if(this.$store.state.vtkContext && 
              this.$store.state.currentUrl !== 
              `http://127.0.0.1:8000/tooth/upper/${currentRow['filename']}`){
                this.$store.dispatch('deleteVtkContext')
                this.$store.dispatch('deleteCentroidsPred')
                this.$store.dispatch('displayCentroids', false)
              }
              //获取新模型的地址
              let currentUrl = `http://127.0.0.1:8000/tooth/upper/${currentRow['filename']}`
              this.$store.dispatch('setCurrentUrl',currentUrl)
            }
            else if (currentRow['tag'] === '下颌'){
              //若vtkContext中有内容，且模型地址发生改变，则清空vtkContext和centroidsPred中的内容
              if(this.$store.state.vtkContext && 
              this.$store.state.currentUrl !== 
              `http://127.0.0.1:8000/tooth/lower/${currentRow['filename']}`){
                this.$store.dispatch('deleteVtkContext')
                this.$store.dispatch('deleteCentroidsPred')
                this.$store.dispatch('displayCentroids', false)
              }
              //获取新模型的地址
              let currentUrl = `http://127.0.0.1:8000/tooth/lower/${currentRow['filename']}`
              this.$store.dispatch('setCurrentUrl',currentUrl)
            }
          }
        }
        },
        mounted(){
          axios.get('http://127.0.0.1:8000/tooth/toothlist').then(
            response => {
              var toothList = JSON.parse(response.data.data)
              toothList['upper'].forEach(element => {
                var toothObj = {
                  filename: element,
                  tag: '上颌'
                }
                this.tableData.unshift(toothObj)
              });
              
              toothList['lower'].forEach(element => {
                var toothObj = {
                  filename: element,
                  tag: '下颌'
                }
                this.tableData.unshift(toothObj)
              });
            },
            error => {
              console.log('error', error.message)
            }
          )
        }
    }
</script>

<style>
  #importButton{
    margin-left: 255px;
    margin-top: 10px;
  }
</style>